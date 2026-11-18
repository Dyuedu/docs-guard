"""Spec 003 AC-3 (shrink map), AC-4 (threshold map), AC-9 (dataset determinism)."""

import numpy as np
import pytest

pytest.importorskip("pyclipper")

from docguard.train.det_labels import make_border_map, make_shrink_map

RECT = np.array([[100, 100], [300, 100], [300, 120], [100, 120]], np.float32)  # 200 x 20
D = 200 * 20 * (1 - 0.4**2) / 440  # AC-3: ~7.64 px


def test_shrink_map_rect():
    gt, mask = make_shrink_map([RECT.copy()], [False], 200, 400)
    assert gt.dtype == np.float32 and mask.dtype == np.float32
    lo, hi_x, hi_y = 100 + D, 300 - D, 120 - D  # inner rectangle, +-1 px at the border
    assert gt[int(np.ceil(lo)) + 1 : int(hi_y), int(np.ceil(lo)) + 1 : int(hi_x)].min() == 1.0
    ys, xs = np.nonzero(gt)
    assert xs.min() >= lo - 1 and xs.max() <= hi_x + 1 and ys.min() >= lo - 1 and ys.max() <= hi_y + 1
    assert mask.min() == 1.0


def test_small_polygon_ignored():
    small = np.array([[50, 50], [120, 50], [120, 56], [50, 56]], np.float32)  # short side 6 < 8
    gt, mask = make_shrink_map([small.copy(), RECT.copy()], [False, True], 200, 400)
    assert gt.max() == 0.0
    assert mask[52, 80] == 0.0 and mask[110, 200] == 0.0  # both polygons masked out of the loss
    assert mask[10, 10] == 1.0


def test_threshold_map_values():
    t, m = make_border_map([RECT.copy()], [False], 200, 400)
    assert t.min() >= 0.3 - 1e-6 and t.max() <= 0.7 + 1e-6
    assert t[100, 200] == pytest.approx(0.7, abs=0.02)  # on the top edge
    assert t[120, 150] == pytest.approx(0.7, abs=0.02)  # on the bottom edge
    assert t[110, 200] == pytest.approx(0.3, abs=1e-6)  # 10 px from both long edges >= D
    assert t[50, 200] == pytest.approx(0.3, abs=1e-6)  # far outside
    assert t[104, 200] == pytest.approx(0.3 + 0.4 * (1 - 4 / D), abs=1e-3)  # AC-4 formula
    assert m[94, 200] == 1.0 and m[110, 200] == 1.0  # inside polygon dilated by D
    assert m[90, 200] == 0.0 and m[50, 200] == 0.0
