"""Intent 002 — synthetic full-page receipts (spec AC-3, AC-6)."""

import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from docguard.data.synth_receipt import DEFAULT_CFG, polygon_iou, render_receipt
from docguard.data.synth_text import ReceiptCorpus, load_vocab_set

TEST_FONT = str(Path(__file__).parent / "fonts" / "BeVietnamPro-Regular.ttf")
VOCAB = Path(__file__).resolve().parents[1] / "configs" / "vocab_vi.txt"
TRAIN_TEXTS = ["MINIMART ANAN", "Chợ Sủi Phú Thị Gia Lâm", "VM+ QNH 590 Trần Phú", "CO.OP MART", "Siêu thị Vinmart"]


@pytest.fixture(scope="module")
def corpus():
    return ReceiptCorpus(TRAIN_TEXTS, load_vocab_set(VOCAB), seed=0)


def _one(corpus, seed, cfg=None):
    rng = random.Random(seed)
    lines = corpus.receipt(rng)
    return render_receipt(lines, [TEST_FONT], rng, cfg or DEFAULT_CFG), lines


def test_polygon_iou_basic():
    a = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert polygon_iou(a, a) == pytest.approx(1.0)
    assert polygon_iou(a, [[20, 20], [30, 20], [30, 30], [20, 30]]) == 0.0
    assert polygon_iou(a, [[5, 0], [15, 0], [15, 10], [5, 10]]) == pytest.approx(1 / 3)


# --- AC-3: >= 8 lines, valid polygons inside the image, no overlap > IoU 0.1 ----------------------

def test_receipt_polygons_valid(corpus):
    for seed in range(20):
        (img, boxes), lines = _one(corpus, seed)
        h, w = img.shape[:2]
        assert img.dtype == np.uint8 and img.shape[2] == 3
        assert len(boxes) >= 8, seed
        assert len(boxes) == len(lines)       # one box per corpus line, in order
        for b, (text, label) in zip(boxes, lines):
            assert b["text"] == text and b["label"] == label
            poly = np.asarray(b["polygon"], dtype=np.float64)
            assert poly.shape == (4, 2)
            assert (poly[:, 0] >= 0).all() and (poly[:, 0] <= w).all(), (seed, poly)
            assert (poly[:, 1] >= 0).all() and (poly[:, 1] <= h).all(), (seed, poly)
            x1, y1, x2, y2 = b["bbox"]
            assert [x1, y1, x2, y2] == [int(poly[:, 0].min()), int(poly[:, 1].min()),
                                        int(np.ceil(poly[:, 0].max())), int(np.ceil(poly[:, 1].max()))]
            assert x2 > x1 and y2 > y1
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                assert polygon_iou(boxes[i]["polygon"], boxes[j]["polygon"]) <= 0.1, (seed, i, j)


def test_receipt_text_height_in_range(corpus):
    # spec AC-4: median text height within [12, 40] px when the long side is 1024
    heights = []
    for seed in range(10):
        (img, boxes), _ = _one(corpus, seed)
        scale = 1024 / max(img.shape[:2])
        for b in boxes:
            p = np.asarray(b["polygon"], dtype=np.float64)
            # height = shorter side of the quad (robust to 90° rotation)
            side_a = np.linalg.norm(p[1] - p[0]); side_b = np.linalg.norm(p[3] - p[0])
            heights.append(min(side_a, side_b) * scale)
    assert 12 <= float(np.median(heights)) <= 40, np.median(heights)


def test_receipt_polygon_matches_ink(corpus):
    # smoke check on the composed image: the first lines' bboxes contain dark ink. The full check
    # (every quad on the paper and inked, 200 seeds) is tests/test_synth_receipt_overflow.py
    cfg = {**DEFAULT_CFG, "rotate90_prob": 0.0, "perspective_prob": 0.0, "elastic_prob": 0.0, "noise_prob": 0.0}
    (img, boxes), _ = _one(corpus, 1, cfg)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    for b in boxes[:5]:
        x1, y1, x2, y2 = b["bbox"]
        inside = gray[y1:y2, x1:x2]
        assert inside.size and inside.min() < 128, b["text"]   # some dark ink inside


# --- AC-6: determinism ----------------------------------------------------------------------------

def test_receipt_deterministic(corpus):
    (a, ba), _ = _one(corpus, 11)
    (b, bb), _ = _one(corpus, 11)
    assert np.array_equal(a, b) and ba == bb


def test_polygon_iou_nearly_parallel_quads_regression():
    # Bug repro (report run 2026-11-12, train images/008245.jpg lines 6,7): two adjacent, slightly
    # tilted line quads that only touch along a thin sliver. cv2.intersectConvexConvex returned the
    # full area of `a` (IoU 0.95). Ground truth is computed independently by rasterising both quads.
    a = [[733.1, 108.0], [717.4, 663.1], [676.6, 663.1], [692.9, 109.1]]
    b = [[692.8, 112.5], [675.2, 713.7], [635.6, 713.6], [653.7, 113.6]]
    s = 8
    ma = np.zeros((800 * s, 800 * s), np.uint8); mb = np.zeros_like(ma)
    cv2.fillPoly(ma, [np.round(np.asarray(a) * s).astype(np.int32)], 1)
    cv2.fillPoly(mb, [np.round(np.asarray(b) * s).astype(np.int32)], 1)
    truth = (ma & mb).sum() / (ma | mb).sum()
    assert truth < 0.01
    assert polygon_iou(a, b) == pytest.approx(truth, abs=0.005)
    assert polygon_iou(b, a) == pytest.approx(truth, abs=0.005)
