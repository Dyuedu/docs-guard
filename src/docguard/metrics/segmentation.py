"""Tampering localization metrics (proposal §7)."""

from __future__ import annotations

import numpy as np


def _binarize(mask: np.ndarray, threshold: float) -> np.ndarray:
    return mask >= threshold if mask.dtype != bool else mask


def pixel_f1(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    """Pixel-level F1 for one image. Both-empty counts as a perfect score."""
    p, g = _binarize(pred, threshold), gt.astype(bool)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    if tp + fp + fn == 0:
        return 1.0
    return float(2 * tp / (2 * tp + fp + fn))


def iou(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> float:
    p, g = _binarize(pred, threshold), gt.astype(bool)
    union = np.logical_or(p, g).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, g).sum() / union)


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Image-level AUC via the Mann-Whitney U statistic (ties get average rank)."""
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both positive and negative samples")
    order = scores.argsort()
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
