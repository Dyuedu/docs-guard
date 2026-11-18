"""End-to-end detection evaluation through the serving path (spec 003 AC-7, AC-10).

image -> P.det_preprocess -> predict_prob -> P.det_postprocess (params from configs/pipeline.yaml)
-> greedy IoU matching against GT boxes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import numpy as np

from docguard.metrics.detection import detection_prf, match_boxes
from docguard.pipeline import processing as P

HEIGHT_BUCKET_FACTOR = 1.6  # spec 003 concern 5: <= 1.6x the field's median height vs taller


def evaluate_det(
    predict_prob: Callable[[np.ndarray], np.ndarray], samples: Iterable[dict], pipeline_cfg: dict,
    labels: set[str] | None = None, iou_thr: float = 0.5,
) -> dict:
    """samples yield {"image": RGB uint8, "boxes": [[x1,y1,x2,y2]], "labels": [str|None]}.

    labels=None scores every GT box; otherwise only GT boxes whose label is in `labels`.
    Returns micro P/R/H-mean, mean IoU of matches, recall per label and per height bucket.
    """
    d = pipeline_cfg["text_det"] if "text_det" in pipeline_cfg else pipeline_cfg
    per_image, ious = [], []
    gt_hit: list[tuple[str | None, float, bool]] = []  # (label, height, matched)
    for s in samples:
        image = s["image"]
        inp, ratio = P.det_preprocess(image, d["input_size"])
        prob = predict_prob(inp)
        preds = P.det_postprocess(prob, ratio, d["bin_threshold"], d["box_threshold"], d["shrink_ratio"],
                                  image.shape[:2])
        keep = [i for i, lb in enumerate(s.get("labels") or [None] * len(s["boxes"]))
                if labels is None or lb in labels]
        gts = [list(s["boxes"][i]) for i in keep]
        gt_labels = [(s.get("labels") or [None] * len(s["boxes"]))[i] for i in keep]
        matches = match_boxes(preds, gts, iou_thr)
        matched_gt = {j for _, j, _ in matches}
        ious.extend(v for _, _, v in matches)
        gt_hit.extend((lb, g[3] - g[1], j in matched_gt) for j, (g, lb) in enumerate(zip(gts, gt_labels)))
        per_image.append((preds, gts))

    out = detection_prf(per_image, iou_thr)
    out["mean_iou"] = float(np.mean(ious)) if ious else 0.0
    out["recall_by_label"], out["recall_by_height"] = _breakdowns(gt_hit)
    return out


def _breakdowns(gt_hit: list[tuple[str | None, float, bool]]) -> tuple[dict, dict]:
    by_label: dict[str, list[bool]] = defaultdict(list)
    heights: dict[str, list[float]] = defaultdict(list)
    for lb, h, hit in gt_hit:
        by_label[str(lb)].append(hit)
        heights[str(lb)].append(h)
    median = {k: float(np.median(v)) for k, v in heights.items()}
    by_height: dict[str, list[bool]] = defaultdict(list)
    for lb, h, hit in gt_hit:
        tall = h > HEIGHT_BUCKET_FACTOR * median[str(lb)]
        by_height[f"{lb}/{'tall' if tall else 'normal'}"].append(hit)
    summarize = lambda d: {k: {"recall": float(np.mean(v)), "n": len(v)} for k, v in sorted(d.items())}
    return summarize(by_label), summarize(by_height)
