import numpy as np
import pytest

from docguard.metrics.ocr import cer, field_exact_match, levenshtein
from docguard.metrics.segmentation import iou, pixel_f1, roc_auc


def test_pixel_f1_and_iou():
    gt = np.zeros((10, 10), bool)
    gt[:5, :5] = True  # 25 px
    pred = np.zeros((10, 10), float)
    pred[:5, :10] = 0.9  # 50 px, 25 overlap
    assert pixel_f1(pred, gt) == pytest.approx(2 * 25 / (2 * 25 + 25))
    assert iou(pred, gt) == pytest.approx(0.5)


def test_both_empty_is_perfect():
    z = np.zeros((4, 4))
    assert pixel_f1(z, z.astype(bool)) == 1.0
    assert iou(z, z.astype(bool)) == 1.0


def test_roc_auc():
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert roc_auc(np.array([0.5, 0.5]), np.array([0, 1])) == 0.5
    with pytest.raises(ValueError):
        roc_auc(np.array([0.1]), np.array([1]))


def test_levenshtein_and_cer():
    assert levenshtein("kitten", "sitting") == 3
    assert cer(["NGUYEN VAN A"], ["NGUYỄN VĂN A"]) == pytest.approx(2 / 12)


def test_field_exact_match():
    preds = [{"total_amount": "125000", "timestamp": "15/08/2021"}, {"total_amount": "99", "timestamp": None}]
    refs = [{"total_amount": "125 000", "timestamp": "15/08/2021"}, {"total_amount": "100", "timestamp": None}]
    em = field_exact_match(preds, refs)
    assert em["total_amount"] == 0.5
    assert em["timestamp"] == 1.0  # None refs are skipped
    assert em["overall"] == pytest.approx(2 / 3)
