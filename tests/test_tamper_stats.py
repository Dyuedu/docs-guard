"""Intent 001 — tamper dataset generation + validation (spec AC-1, AC-2, AC-4, AC-5).

Expected values come from intent/001-tamper-dataset/spec.md, not from running the code.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from docguard.data.annotations import Sample, TextBox
from docguard.data.tamper_stats import leak_check, load_index, mask_stats, preview_grid, summarize
from scripts.make_tamper_dataset import generate, resolve_fonts

_FONTS = [p for p in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if Path(p).exists()]
needs_font = pytest.mark.skipif(not _FONTS, reason="no TTF font available")


def _make_receipts(img_dir: Path, n: int) -> list[Sample]:
    img_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for k in range(n):
        rng = np.random.default_rng(k)
        img = np.full((400, 300, 3), 245, np.uint8)
        boxes = []
        for y in (50, 150, 250):
            txt = f"{rng.integers(10_000, 999_999):,}"
            cv2.putText(img, txt, (20, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 2)
            boxes.append(TextBox([20, y, 220, y + 35], txt, "TOTAL_COST"))
        name = f"r{k}.jpg"
        cv2.imwrite(str(img_dir / name), img)
        samples.append(Sample(name, 400, 300, boxes))
    return samples


SPLITS = {"train": {"r0.jpg", "r1.jpg", "r2.jpg", "r3.jpg"}, "val": {"r4.jpg"}, "test": {"r5.jpg"}}
PER_IMAGE = {"train": 4, "val": 2, "test": 2}


@pytest.fixture
def generated(tmp_path):
    samples = _make_receipts(tmp_path / "imgs", 6)
    out = tmp_path / "tamper"
    index = generate(samples, SPLITS, tmp_path / "imgs", out, _FONTS, PER_IMAGE, seed=42, log=lambda *_: None)
    return out, index


# --- AC-5: determinism -------------------------------------------------------------------------

@needs_font
def test_generation_deterministic(tmp_path, generated):
    out1, _ = generated
    samples = _make_receipts(tmp_path / "imgs2", 6)
    out2 = tmp_path / "tamper2"
    generate(samples, SPLITS, tmp_path / "imgs2", out2, _FONTS, PER_IMAGE, seed=42, log=lambda *_: None)
    for split in SPLITS:
        assert (out1 / split / "index.jsonl").read_bytes() == (out2 / split / "index.jsonl").read_bytes()


# --- generation contract (spec: index.jsonl fields, per-split counts) --------------------------

@needs_font
def test_generated_index_contract(generated):
    out, index = generated
    for split, rows in index.items():
        assert rows == load_index(out / split)
        auth = [r for r in rows if r["op"] == "authentic"]
        fake = [r for r in rows if r["tampered"] == 1]
        assert len(auth) == len(SPLITS[split])                       # one authentic per receipt
        assert len(fake) <= PER_IMAGE[split] * len(SPLITS[split])    # at most N tampered per receipt
        assert all(70 <= r["jpeg_q"] <= 95 for r in rows)
        for r in fake:
            assert r["op"] in {"copy_move", "splice", "inpaint"}
            if r["op"] == "splice":
                assert r["donor"] in SPLITS[split]                   # donor from the same split
            if r["op"] == "inpaint":
                assert r["font"] in {Path(f).name for f in _FONTS}


def test_resolve_fonts(tmp_path):
    (tmp_path / "b.ttf").write_bytes(b"x")
    (tmp_path / "a.ttf").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x")
    assert [Path(p).name for p in resolve_fonts(tmp_path)] == ["a.ttf", "b.ttf"]
    assert resolve_fonts(tmp_path / "a.ttf") == [str(tmp_path / "a.ttf")]
    with pytest.raises(FileNotFoundError):
        resolve_fonts(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_fonts(empty)


# --- AC-1: leak check ---------------------------------------------------------------------------

@needs_font
def test_leak_check_passes_on_generated(generated):
    _, index = generated
    assert leak_check(index, SPLITS) == []


def _row(name, source, split_op="authentic", **kw):
    return {"name": name, "source": source, "op": split_op, "tampered": int(split_op != "authentic"), "jpeg_q": 80, **kw}


def test_leak_check_detects_source_in_two_splits():
    index = {"train": [_row("a", "r0.jpg")], "val": [_row("b", "r0.jpg")], "test": []}
    splits = {"train": {"r0.jpg"}, "val": {"r4.jpg"}, "test": set()}
    assert leak_check(index, splits)


def test_leak_check_detects_source_outside_canonical_split():
    index = {"train": [_row("a", "r4.jpg")], "val": [], "test": []}
    assert leak_check(index, SPLITS)


def test_leak_check_detects_cross_split_donor():
    index = {"train": [_row("a", "r0.jpg", "splice", donor="r5.jpg")], "val": [], "test": []}
    assert leak_check(index, SPLITS)


# --- AC-2: mask validity ------------------------------------------------------------------------

@needs_font
def test_mask_stats_valid_on_generated(generated):
    out, index = generated
    for split, rows in index.items():
        stats = mask_stats(out / split, rows)
        assert len(stats) == len(rows)
        assert all(s["valid"] for s in stats), [s for s in stats if not s["valid"]]
        for s, r in zip(stats, rows):
            assert (s["area_ratio"] == 0) == (r["tampered"] == 0)


def _write_pair(d: Path, name: str, img_hw, mask: np.ndarray | None):
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(d / "images" / f"{name}.jpg"), np.full((*img_hw, 3), 200, np.uint8))
    if mask is not None:
        cv2.imwrite(str(d / "masks" / f"{name}.png"), mask)


def test_mask_stats_flags_each_invalid_case(tmp_path):
    good = np.zeros((40, 40), np.uint8)
    good[5:10, 5:10] = 255
    nonbinary = good.copy()
    nonbinary[0, 0] = 128
    authentic_dirty = np.zeros((40, 40), np.uint8)
    authentic_dirty[0, 0] = 255
    cases = [
        ("ok", (40, 40), good, "copy_move", True),
        ("shape", (40, 50), good, "copy_move", False),          # image/mask H×W differ
        ("nonbin", (40, 40), nonbinary, "splice", False),       # values outside {0,255}
        ("empty", (40, 40), np.zeros((40, 40), np.uint8), "inpaint", False),  # tampered but empty mask
        ("dirty", (40, 40), authentic_dirty, "authentic", False),  # authentic but non-zero mask
        ("nomask", (40, 40), None, "copy_move", False),         # missing mask file
    ]
    rows = []
    for name, hw, mask, op, _ in cases:
        _write_pair(tmp_path, name, hw, mask)
        rows.append(_row(name, f"{name}.jpg", op))
    stats = {s["name"]: s for s in mask_stats(tmp_path, rows)}
    for name, *_, expect_valid in cases:
        assert stats[name]["valid"] is expect_valid, (name, stats[name])
        if not expect_valid:
            assert stats[name]["error"]
    assert stats["ok"]["area_ratio"] == pytest.approx(25 / 1600)


# --- AC-3 / AC-4: summary checks ----------------------------------------------------------------

def _stat(name, op, ratio):
    return {"name": name, "op": op, "h": 100, "w": 100, "area_ratio": ratio, "valid": True, "error": ""}


def _summary_for(ops_ratios):
    rows = [_row(f"x{i}", f"r{i}.jpg", op) for i, (op, _) in enumerate(ops_ratios)]
    stats = [_stat(f"x{i}", op, r) for i, (op, r) in enumerate(ops_ratios)]
    return summarize({"train": rows, "val": [], "test": []}, {"train": stats, "val": [], "test": []}, leak_errors=[])


def test_summarize_passes_balanced_and_in_range():
    s = _summary_for([("copy_move", 0.001), ("splice", 0.002), ("inpaint", 0.003), ("authentic", 0.0)])
    assert s["checks"] == {"no_leak": True, "masks_valid": True, "op_balance": True, "mask_area": True}
    assert s["splits"]["train"]["op_ratio"]["copy_move"] == pytest.approx(1 / 3)


def test_summarize_fails_op_below_20_percent():
    # copy_move = 1/6 ≈ 16.7% < 20% (spec AC-3)
    ops = [("copy_move", 0.001)] + [("splice", 0.001)] * 3 + [("inpaint", 0.001)] * 2
    assert _summary_for(ops)["checks"]["op_balance"] is False


@pytest.mark.parametrize("ratio, ok", [(0.00005, False), (0.0001, True), (0.01, True), (0.02, False)])
def test_summarize_median_area_bounds(ratio, ok):
    # spec AC-4: median tampered area ratio must lie in [0.0001, 0.01]
    s = _summary_for([("copy_move", ratio), ("splice", ratio), ("inpaint", ratio)])
    assert s["checks"]["mask_area"] is ok


def test_summarize_flags_outliers_above_5_percent():
    s = _summary_for([("copy_move", 0.001), ("splice", 0.06), ("inpaint", 0.002)])
    assert [o["name"] for o in s["outliers"]] == ["x1"]
    assert s["checks"]["mask_area"] is True   # outliers are flagged for review, not a failure


def test_summarize_propagates_leak_and_invalid_masks():
    rows = [_row("x0", "r0.jpg", "copy_move")]
    bad = [dict(_stat("x0", "copy_move", 0.0), valid=False, error="empty mask")]
    s = summarize({"train": rows, "val": [], "test": []}, {"train": bad, "val": [], "test": []}, leak_errors=["leak!"])
    assert s["checks"]["no_leak"] is False
    assert s["checks"]["masks_valid"] is False


# --- preview -----------------------------------------------------------------------------------

@needs_font
def test_preview_grid_writes_image(tmp_path, generated):
    out, index = generated
    fake = [r for r in index["train"] if r["tampered"]][:3]
    dst = tmp_path / "prev" / "grid.jpg"
    preview_grid(out / "train", fake, dst)
    img = cv2.imread(str(dst))
    assert img is not None and img.shape[0] > 0
