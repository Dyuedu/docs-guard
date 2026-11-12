"""Bug repro (verifier major #1, intent 002): blank-line separators are not counted in the paper
height, so the last lines can be drawn past the bottom edge of the paper while their polygons are
still exported. Spec AC-3: every exported line has a polygon over its text, so every quad must lie
on the paper and contain ink.
"""

import random
from pathlib import Path

import numpy as np

from docguard.data.synth_receipt import DEFAULT_CFG, _render_paper

TEST_FONT = str(Path(__file__).parent / "fonts" / "BeVietnamPro-Regular.ttf")
# 30 short lines -> narrow paper, small margin, many chances (6 % each) for a blank separator
LINES = [("CO.OP MART", "SELLER")] + [(f"Món {i}   12.000", "OTHER") for i in range(29)]


def test_every_line_quad_lies_on_paper_and_has_ink():
    for seed in range(200):
        rng = random.Random(seed)
        paper, alpha, quads = _render_paper(LINES, [TEST_FONT], rng, DEFAULT_CFG, text_px=24)
        h, w = alpha.shape
        gray = paper.mean(axis=2)
        for k, q in enumerate(quads):
            x1, y1 = np.floor(q.min(axis=0)).astype(int)
            x2, y2 = np.ceil(q.max(axis=0)).astype(int)
            assert x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h, (seed, k, (x1, y1, x2, y2), (w, h))
            assert gray[y1:y2, x1:x2].min() < 160, (seed, k)   # the line's ink is actually on the paper
