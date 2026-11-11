import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from docguard.data.annotations import TextBox
from docguard.data.synth_tamper import copy_move, digit_boxes, inpaint_and_render, jpeg_recompress, splice


def _receipt(seed: int):
    rng = np.random.default_rng(seed)
    img = np.full((400, 300, 3), 245, np.uint8)
    boxes = []
    for i, y in enumerate((50, 150, 250)):
        txt = f"{rng.integers(10_000, 999_999):,}"
        cv2.putText(img, txt, (20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
        boxes.append(TextBox([20, y, 220, y + 35], txt, "TOTAL_COST" if i == 2 else "OTHER"))
    boxes.append(TextBox([20, 320, 200, 350], "CAM ON QUY KHACH"))
    return img, boxes


def test_digit_boxes_filters_non_numeric():
    _, boxes = _receipt(0)
    assert len(digit_boxes(boxes)) == 3


def test_copy_move_mask_matches_changed_pixels():
    img, boxes = _receipt(0)
    out, mask = copy_move(img, boxes, random.Random(1))
    changed = np.any(out != img, axis=-1)
    assert mask.any()
    assert np.array_equal(changed, mask > 0)


def test_splice_only_changes_masked_region():
    img, boxes = _receipt(0)
    donor, dboxes = _receipt(1)
    out, mask = splice(img, boxes, donor, dboxes, random.Random(2))
    assert np.array_equal(out[mask == 0], img[mask == 0])


def test_copy_move_needs_two_digit_boxes():
    img, boxes = _receipt(0)
    with pytest.raises(ValueError):
        copy_move(img, boxes[:1], random.Random(0))


_FONTS = [p for p in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(p).exists()]


@pytest.mark.skipif(not _FONTS, reason="no TTF font available")
def test_inpaint_and_render_mask_matches_changed_pixels():
    img, boxes = _receipt(0)
    out, mask, new_text = inpaint_and_render(img, boxes, _FONTS[0], random.Random(3))
    assert any(c.isdigit() for c in new_text)
    assert np.array_equal(np.any(out != img, axis=-1), mask > 0)


def test_jpeg_recompress_keeps_shape():
    img, _ = _receipt(0)
    assert jpeg_recompress(img, 75).shape == img.shape


# --- intent 001 AC-8: inpaint must stay inside the chosen box; vertical boxes are not inpainted ---

def _single_box_receipt(box, text):
    img = np.full((400, 300, 3), 245, np.uint8)
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), (20, 20, 20), -1)  # dark "ink" inside the box
    return img, [TextBox(list(box), text, "TOTAL_COST")]


@pytest.mark.skipif(not _FONTS, reason="no TTF font available")
def test_inpaint_changes_only_pixels_inside_box():
    # narrow, tall-ish horizontal box: font size h*0.8 would overflow its width
    box = (20, 50, 90, 100)
    img, boxes = _single_box_receipt(box, "125,000")
    for seed in range(5):
        out, mask, _ = inpaint_and_render(img, boxes, _FONTS[0], random.Random(seed))
        ys, xs = np.nonzero(mask)
        assert len(xs)
        assert xs.min() >= box[0] and xs.max() < box[2] and ys.min() >= box[1] and ys.max() < box[3], (seed, xs.min(), xs.max(), ys.min(), ys.max())


@pytest.mark.skipif(not _FONTS, reason="no TTF font available")
def test_inpaint_rejects_vertical_box():
    img, boxes = _single_box_receipt((100, 20, 140, 300), "18/08/2020")  # h=280 > w=40: rotated receipt
    with pytest.raises(ValueError):
        inpaint_and_render(img, boxes, _FONTS[0], random.Random(0))
