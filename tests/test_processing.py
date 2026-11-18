import cv2
import numpy as np
import pytest

from docguard.pipeline import processing as P


def test_tile_stitch_roundtrip_covers_every_pixel():
    img = np.random.randint(0, 255, (700, 1100, 3), np.uint8)
    patches, coords = P.tile(img, 512, 384)
    assert patches.shape[1:] == (3, 512, 512)
    ones = np.ones((len(coords), 512, 512), np.float32)
    assert np.allclose(P.stitch(ones, coords, img.shape[:2], 512), 1.0)


def test_tile_small_image_is_padded():
    img = np.zeros((100, 200, 3), np.uint8)
    patches, coords = P.tile(img, 512, 384)
    assert coords == [(0, 0)] and patches.shape == (1, 3, 512, 512)
    assert P.stitch(np.ones((1, 512, 512)), coords, (100, 200), 512).shape == (100, 200)


def test_mask_to_boxes_filters_small_regions():
    prob = np.zeros((100, 100), np.float32)
    prob[10:30, 20:60] = 0.9
    prob[80, 80] = 0.9  # 1 px noise
    assert P.mask_to_boxes(prob, 0.5, min_area=16) == [[20, 10, 60, 30]]


def test_det_postprocess_maps_back_to_image_coords():
    image = np.zeros((480, 960, 3), np.uint8)
    inp, ratio = P.det_preprocess(image, 960)
    assert inp.shape == (1, 3, 960, 960) and ratio == 1.0
    prob = np.zeros((960, 960), np.float32)
    prob[100:120, 200:400] = 0.95
    boxes = P.det_postprocess(prob, ratio, 0.3, 0.6, 0.4, (480, 960))
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 < 200 and y1 < 100 and x2 > 400 and y2 > 120  # unclipped outward


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def _gt_kernel(prob, box, r=0.4):
    """Spec 003 AC-3: shrink by D = A(1 - r^2)/L on every side, rasterised with fillPoly."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    d = w * h * (1 - r * r) / (2 * (w + h))
    pts = np.array([[x1 + d, y1 + d], [x2 - d, y1 + d], [x2 - d, y2 - d], [x1 + d, y2 - d]])
    cv2.fillPoly(prob, [np.round(pts).astype(np.int32)], 1.0)


def test_det_postprocess_exact_unclip_recovers_gt_box():
    # Spec 003 concern 1A / AC-7: GT shrink kernel of 20x200 and 30x600 lines -> each IoU >= 0.85 and
    # mean >= 0.85 (plan 003 Deviations: 0.95 is unreachable once the kernel is rasterised).
    gts = [[100, 100, 300, 120], [100, 300, 700, 330]]
    prob = np.zeros((960, 960), np.float32)
    for g in gts:
        _gt_kernel(prob, g)
    boxes = P.det_postprocess(prob, 1.0, 0.3, 0.6, 0.4, (960, 960))
    assert len(boxes) == 2
    ious = [_iou(b, g) for b, g in zip(boxes, gts)]
    assert min(ious) >= 0.85 and np.mean(ious) >= 0.85, (boxes, ious)


@pytest.mark.parametrize("deg", [3.0, -5.0])
def test_det_postprocess_skewed_line_recovers_gt_bbox(deg):
    # Plan 003 D8: synth/receipt lines are skewed a few degrees. GT kernel = the w x h rotated rect
    # shrunk by D = A(1 - r^2)/L per side; the box must match the GT polygon's axis-aligned bbox.
    r, (cx, cy), (w, h) = 0.4, (480.0, 300.0), (300.0, 30.0)
    d = w * h * (1 - r * r) / (2 * (w + h))
    gt_poly = cv2.boxPoints(((cx, cy), (w, h), deg))
    gt = [*gt_poly.min(0), *gt_poly.max(0)]
    prob = np.zeros((960, 960), np.float32)
    kernel = cv2.boxPoints(((cx, cy), (w - 2 * d, h - 2 * d), deg))
    cv2.fillPoly(prob, [np.round(kernel).astype(np.int32)], 1.0)
    boxes = P.det_postprocess(prob, 1.0, 0.3, 0.6, r, (960, 960))
    assert len(boxes) == 1
    assert _iou(boxes[0], gt) >= 0.85, (boxes, gt)


def test_det_postprocess_keeps_two_pixel_kernel():
    # Spec 003 Flagged concern 1A: kernel filter is `< 2` px, so a 2 px tall kernel survives.
    prob = np.zeros((960, 960), np.float32)
    prob[100:102, 200:260] = 0.9
    assert len(P.det_postprocess(prob, 1.0, 0.3, 0.6, 0.4, (960, 960))) == 1


def test_crop_for_rec_shape_and_padding():
    img = np.full((100, 300, 3), 255, np.uint8)
    out = P.crop_for_rec(img, [[0, 0, 64, 32], [0, 0, 300, 20]], 32, 320)
    assert out.shape == (2, 3, 32, 320)
    assert np.all(out[0, :, :, 64:] == 0)  # right-padded after resized width 64


def test_ctc_greedy_decode_collapses_repeats_and_blanks():
    vocab = list("ab")
    seq = [1, 1, 0, 1, 2, 2, 0]  # a a _ a b b _  -> "aab"
    logits = np.full((1, len(seq), 3), -10.0, np.float32)
    for t, k in enumerate(seq):
        logits[0, t, k] = 10.0
    (text, conf), = P.ctc_greedy_decode(logits, vocab)
    assert text == "aab" and conf > 0.99


def test_load_vocab_space_token(tmp_path):
    p = tmp_path / "v.txt"
    p.write_text("<space>\na\nb\n", encoding="utf-8")
    assert P.load_vocab(str(p)) == [" ", "a", "b"]


def test_decode_image_downscales_large_input():
    img = np.zeros((3000, 1500, 3), np.uint8)
    ok, buf = cv2.imencode(".png", img)
    out, scale = P.decode_image(buf.tobytes(), 2048)
    assert max(out.shape[:2]) == 2048 and abs(scale - 2048 / 3000) < 1e-6


def test_decide():
    cfg = {"tampered_area_ratio": 0.002, "tampered_score": 0.8}
    assert P.decide(0.1, 0.0, cfg)[0] is True
    assert P.decide(0.1, 0.01, cfg)[0] is False
    assert P.decide(0.95, 0.0, cfg)[0] is False
