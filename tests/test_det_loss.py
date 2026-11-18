"""Spec 003 AC-5: L = 5*BCE_ohem + 10*L1(thresh) + 1*Dice(binary)."""

import math

import pytest

torch = pytest.importorskip("torch")

from docguard.train.det_loss import DBLoss


def _labels(n=2, s=32):
    gt = torch.zeros(n, s, s)
    gt[:, 8:16, 4:28] = 1
    th = torch.full((n, s, s), 0.3)
    th[:, 6:18, 2:30] = 0.6
    th_mask = torch.zeros(n, s, s)
    th_mask[:, 4:20, 0:32] = 1
    return {"shrink_map": gt, "shrink_mask": torch.ones(n, s, s), "thresh_map": th, "thresh_mask": th_mask}


def test_perfect_prediction_near_zero():
    # AC-5 (a): logits +-20, thresh = label -> L < 1e-3.
    b = _labels()
    gt = b["shrink_map"].unsqueeze(1)
    out = {"prob_logits": gt * 40 - 20, "thresh": b["thresh_map"].unsqueeze(1), "binary": gt.clone()}
    losses = DBLoss()(out, b)
    assert set(losses) == {"loss", "loss_prob", "loss_thresh", "loss_binary"}
    assert float(losses["loss"]) < 1e-3


def _bce(z, y):
    return math.log1p(math.exp(-z)) if y == 1 else math.log1p(math.exp(z))


def test_ohem_counts_three_hardest_negatives():
    # AC-5 (b): 1 positive, 10 negatives -> exactly the 3 highest-loss negatives are counted.
    pos_z = 0.5
    neg_z = [-3.0, -2.0, -1.0, 0.0, 0.2, 0.4, 1.0, 1.5, 2.0, 2.5]
    logits = torch.tensor([pos_z] + neg_z).reshape(1, 1, 1, 11)
    gt = torch.zeros(1, 1, 11)
    gt[0, 0, 0] = 1
    b = {"shrink_map": gt, "shrink_mask": torch.ones(1, 1, 11),
         "thresh_map": torch.zeros(1, 1, 11), "thresh_mask": torch.zeros(1, 1, 11)}
    out = {"prob_logits": logits, "thresh": torch.zeros(1, 1, 1, 11), "binary": torch.zeros(1, 1, 1, 11)}
    expected = (_bce(pos_z, 1) + sum(_bce(z, 0) for z in (2.5, 2.0, 1.5))) / 4
    assert float(DBLoss()(out, b)["loss_prob"]) == pytest.approx(expected, rel=1e-5)


def test_no_positive_gives_zero_prob_loss():
    b = _labels()
    b["shrink_map"].zero_()
    out = {"prob_logits": torch.randn(2, 1, 32, 32), "thresh": torch.rand(2, 1, 32, 32), "binary": torch.rand(2, 1, 32, 32)}
    assert float(DBLoss()(out, b)["loss_prob"]) == 0.0


def test_finite_under_fp16():
    # AC-5 (c): fp16 predictions as produced by autocast (on GPU when available).
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = {k: v.to(dev) for k, v in _labels().items()}
    logits = (torch.randn(2, 1, 32, 32, device=dev) * 30).half().requires_grad_()
    thresh = torch.rand(2, 1, 32, 32, device=dev).half()
    with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        binary = torch.sigmoid(50 * (torch.sigmoid(logits) - thresh))
        losses = DBLoss()({"prob_logits": logits, "thresh": thresh, "binary": binary}, b)
    assert all(torch.isfinite(v).item() for v in losses.values())
    losses["loss"].backward()
    assert torch.isfinite(logits.grad).all()
