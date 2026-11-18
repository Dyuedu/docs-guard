"""Text detection metrics (spec 003 AC-6): greedy 1-1 IoU matching on axis-aligned boxes."""

from __future__ import annotations


def box_iou(a: list[int], b: list[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_boxes(preds: list[list[int]], gts: list[list[int]], iou_thr: float = 0.5) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU; a pair matches when IoU >= iou_thr."""
    pairs = sorted(
        ((box_iou(p, g), i, j) for i, p in enumerate(preds) for j, g in enumerate(gts)),
        key=lambda t: -t[0],
    )
    used_p, used_g, out = set(), set(), []
    for v, i, j in pairs:
        if v < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        out.append((i, j, v))
    return out


def detection_prf(per_image: list[tuple[list[list[int]], list[list[int]]]], iou_thr: float = 0.5) -> dict[str, float]:
    """Micro-averaged precision / recall / H-mean over (preds, gts) per image."""
    n_pred = n_gt = n_match = 0
    for preds, gts in per_image:
        n_pred += len(preds)
        n_gt += len(gts)
        n_match += len(match_boxes(preds, gts, iou_thr))
    p = n_match / n_pred if n_pred else 0.0
    r = n_match / n_gt if n_gt else 0.0
    h = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "hmean": h, "n_pred": n_pred, "n_gt": n_gt, "n_match": n_match}
