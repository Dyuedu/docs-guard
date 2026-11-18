"""DB loss: alpha * balanced-BCE(prob logits, OHEM 3:1) + beta * masked L1(thresh) + Dice(binary).

Follows MMOCR (Apache-2.0):
  mmocr/models/textdet/module_losses/db_module_loss.py   (DBModuleLoss)
  mmocr/models/common/losses/bce_loss.py                  (MaskedBalancedBCEWithLogitsLoss)
  mmocr/models/common/losses/dice_loss.py, l1_loss.py     (masked Dice / L1)
Changes: single module, batch tensors instead of data samples; every term computed in fp32 so the
loss is finite under fp16 autocast. fallback_negative_num = 0 (MMOCR default). See NOTICE.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def balanced_bce_with_logits(logits: Tensor, gt: Tensor, mask: Tensor, ohem_ratio: float, eps: float) -> Tensor:
    """All positives + the k_neg = min(#neg, ratio * #pos) hardest negatives, counted over the batch."""
    logits, gt, mask = logits.float(), gt.float(), mask.float()
    positive = gt * mask
    negative = (1 - gt) * mask
    n_pos = int(positive.sum())
    n_neg = min(int(negative.sum()), int(n_pos * ohem_ratio)) if n_pos > 0 else 0
    if n_pos == 0:
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    neg_loss, _ = torch.topk((loss * negative).reshape(-1), n_neg)
    return ((loss * positive).sum() + neg_loss.sum()) / (n_pos + n_neg + eps)


def masked_l1(pred: Tensor, gt: Tensor, mask: Tensor, eps: float) -> Tensor:
    pred, gt, mask = pred.float(), gt.float(), mask.float()
    return (torch.abs(pred - gt) * mask).sum() / (mask.sum() + eps)


def masked_dice(pred: Tensor, gt: Tensor, mask: Tensor, eps: float) -> Tensor:
    pred, gt, mask = pred.float(), gt.float(), mask.float()
    inter = (pred * gt * mask).sum()
    union = (pred * mask).sum() + (gt * mask).sum() + eps
    return 1 - 2 * inter / union


class DBLoss(nn.Module):
    def __init__(self, alpha: float = 5.0, beta: float = 10.0, ohem_ratio: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.ohem_ratio, self.eps = alpha, beta, ohem_ratio, eps

    def forward(self, out: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, Tensor]:
        def m(k: str) -> Tensor:  # label maps are (N,H,W); predictions are (N,1,H,W)
            v = batch[k]
            return v.unsqueeze(1) if v.dim() == 3 else v

        gt, gt_mask, th, th_mask = m("shrink_map"), m("shrink_mask"), m("thresh_map"), m("thresh_mask")
        loss_prob = balanced_bce_with_logits(out["prob_logits"], gt, gt_mask, self.ohem_ratio, self.eps)
        loss_thresh = masked_l1(out["thresh"], th, th_mask, self.eps)
        loss_binary = masked_dice(out["binary"], gt, gt_mask, self.eps)
        loss = self.alpha * loss_prob + self.beta * loss_thresh + loss_binary
        return {"loss": loss, "loss_prob": loss_prob, "loss_thresh": loss_thresh, "loss_binary": loss_binary}
