"""DBNet: torchvision MobileNetV3-large + DB FPN neck + DB head (spec 003).

Neck and head ported from PaddleOCR (Apache-2.0):
  ppocr/modeling/necks/db_fpn.py       (DBFPN, use_asf=False)
  ppocr/modeling/heads/det_db_head.py  (Head, DBHead)
Changes: PyTorch instead of Paddle; the probability head returns logits and sigmoid is applied
only in eval() (safe BCE under AMP); eval() returns a single `prob` tensor for ONNX export and
skips the threshold branch. Backbone from torchvision (BSD-3). See NOTICE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

# End of features[i] -> strides 4, 8, 16, 32 (24, 40, 112, 960 channels).
FEATURE_IDX = (3, 6, 12, 16)
FEATURE_CH = (24, 40, 112, 960)


class Backbone(nn.Module):
    def __init__(self, pretrained: bool):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.features = mobilenet_v3_large(weights=weights).features[: FEATURE_IDX[-1] + 1]

    def forward(self, x: Tensor) -> list[Tensor]:
        outs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in FEATURE_IDX:
                outs.append(x)
        return outs


class DBFPN(nn.Module):
    def __init__(self, in_channels: tuple[int, ...], out_channels: int):
        super().__init__()
        self.ins = nn.ModuleList(nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels)
        self.ps = nn.ModuleList(
            nn.Conv2d(out_channels, out_channels // 4, 3, padding=1, bias=False) for _ in in_channels
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, feats: list[Tensor]) -> Tensor:
        in2, in3, in4, in5 = (conv(f) for conv, f in zip(self.ins, feats))
        out4 = in4 + F.interpolate(in5, scale_factor=2.0, mode="nearest")  # 1/16
        out3 = in3 + F.interpolate(out4, scale_factor=2.0, mode="nearest")  # 1/8
        out2 = in2 + F.interpolate(out3, scale_factor=2.0, mode="nearest")  # 1/4
        p2, p3, p4, p5 = (conv(f) for conv, f in zip(self.ps, (out2, out3, out4, in5)))
        p5 = F.interpolate(p5, scale_factor=8.0, mode="nearest")
        p4 = F.interpolate(p4, scale_factor=4.0, mode="nearest")
        p3 = F.interpolate(p3, scale_factor=2.0, mode="nearest")
        return torch.cat([p5, p4, p3, p2], dim=1)


def _bn(c: int) -> nn.BatchNorm2d:
    bn = nn.BatchNorm2d(c)
    nn.init.constant_(bn.weight, 1.0)
    nn.init.constant_(bn.bias, 1e-4)
    return bn


class Head(nn.Module):
    """conv3x3-BN-ReLU -> deconv2x2 s2-BN-ReLU -> deconv2x2 s2 (stride 4 -> full resolution)."""

    def __init__(self, in_channels: int):
        super().__init__()
        mid = in_channels // 4
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1, bias=False), _bn(mid), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, mid, 2, stride=2), _bn(mid), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid, 1, 2, stride=2),
        )
        for m in self.body:
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_uniform_(m.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)  # logits


class DBNet(nn.Module):
    def __init__(self, pretrained: bool = True, inner_channels: int = 256, k: float = 50.0):
        super().__init__()
        self.k = k
        self.backbone = Backbone(pretrained)
        self.neck = DBFPN(FEATURE_CH, inner_channels)
        self.prob_head = Head(inner_channels)
        self.thresh_head = Head(inner_channels)

    def forward(self, x: Tensor) -> dict[str, Tensor] | Tensor:
        fuse = self.neck(self.backbone(x))
        prob_logits = self.prob_head(fuse)
        if not self.training:
            return torch.sigmoid(prob_logits)
        thresh = torch.sigmoid(self.thresh_head(fuse))
        binary = torch.sigmoid(self.k * (torch.sigmoid(prob_logits) - thresh))
        return {"prob_logits": prob_logits, "thresh": thresh, "binary": binary}
