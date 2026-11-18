"""Spec 003 AC-6: detection matching and micro P/R/H-mean."""

import pytest

from docguard.metrics.detection import box_iou, detection_prf, match_boxes


def test_box_iou_reference_case():
    # AC-6 (a): intersection 5320, union 7080.
    assert box_iou([100, 50, 300, 80], [110, 52, 310, 84]) == pytest.approx(5320 / 7080)
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0


def test_prf_micro_aggregate():
    # AC-6 (b): 20 GT, 22 pred, 19 matched -> P 0.8636, R 0.95, H 0.9048.
    gts = [[i * 50, 0, i * 50 + 40, 20] for i in range(20)]
    preds = [list(g) for g in gts[:19]] + [[2000 + i * 50, 500, 2040 + i * 50, 520] for i in range(3)]
    m = detection_prf([(preds[:10], gts[:10]), (preds[10:], gts[10:])])
    assert (m["n_pred"], m["n_gt"], m["n_match"]) == (22, 20, 19)
    assert m["precision"] == pytest.approx(0.8636, abs=1e-4)
    assert m["recall"] == pytest.approx(0.95)
    assert m["hmean"] == pytest.approx(0.9048, abs=1e-4)


def test_one_pred_covering_two_gts_matches_once():
    # AC-6 (c)
    gts = [[0, 0, 100, 20], [0, 22, 100, 42]]
    preds = [[0, 0, 100, 42]]
    assert len(match_boxes(preds, gts, 0.4)) == 1


def test_greedy_by_descending_iou_and_threshold():
    gts = [[0, 0, 100, 20]]
    preds = [[0, 0, 100, 24], [0, 0, 100, 20]]
    assert match_boxes(preds, gts) == [(1, 0, 1.0)]
    assert match_boxes([[0, 0, 100, 20]], [[0, 0, 100, 41]]) == []  # IoU 20/41 < 0.5


def test_zero_denominators():
    m = detection_prf([([], [])])
    assert m["precision"] == m["recall"] == m["hmean"] == 0.0
