"""Validate the synthetic OCR data and write a report (intent 002, AC-1..AC-5, AC-7, AC-8).

    python scripts/synth_ocr_report.py --root data/synth_ocr --splits data/processed/splits.json \
        --ann data/processed/annotations.jsonl --out results/data/synth_ocr --preview 30

Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from docguard.data.annotations import read_jsonl
from docguard.data.splits import load_splits
from docguard.data.synth_receipt import polygon_iou
from docguard.data.synth_text import LineRenderer, ReceiptCorpus, filter_charset, heldout_only, load_vocab_set

# Thresholds from intent/002-synth-ocr-data/spec.md
MIN_LINES, MAX_IOU = 8, 0.1
HEIGHT_RANGE = (12, 40)          # px at long side 1024 (AC-4)
LINE_H, LINE_MAX_W, LINE_MAX_LEN = 32, 320, 40
LINE_MS_BUDGET = 5.0             # AC-7


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_index(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def check_charset(texts, vocab) -> list[str]:
    allowed = vocab | {" "}
    return sorted({c for t in texts for c in t if c not in allowed})


def check_provenance(corpus_lines: list[str], train_texts: list[str], heldout_texts: list[str]) -> tuple[list[str], dict]:
    """AC-2 (amended): every corpus line must come from train texts. Lines that also occur in
    val/test are legitimate. Returns (lines not sourced from train, info with held-out overlap)."""
    norm = ReceiptCorpus.normalize
    train = {norm(t) for t in train_texts}
    leaked = sorted({l for l in corpus_lines if norm(l) not in train})
    held = [norm(t) for t in heldout_texts if len(t.strip()) > 3]
    overlap = sum(h in train for h in held)
    return leaked, {"heldout_texts": len(held), "heldout_in_train": overlap,
                    "heldout_overlap_ratio": overlap / len(held) if held else 0.0}


def check_heldout_generated(generated_texts: list[str], train_texts: list[str], heldout_texts: list[str]) -> list[str]:
    """AC-2 clause 2: generated texts (receipts + lines) must not reproduce val/test-only text
    (> 3 chars after normalisation). Returns the offending normalised texts, sorted, unique."""
    banned = heldout_only(train_texts, heldout_texts)
    return sorted({n for n in (ReceiptCorpus.normalize(t) for t in generated_texts) if n in banned})


def _split_texts(ann_path: Path, splits: dict, vocab: set[str]) -> tuple[list[str], list[str]]:
    train, held = [], []
    for s in read_jsonl(ann_path):
        dst = train if s.image in splits["train"] else held if s.image in splits["val"] | splits["test"] else None
        if dst is not None:
            dst += [" ".join(filter_charset(b.text, vocab).split()) for b in s.boxes]
    return train, held


def check_receipts(split_dir: Path, rows: list[dict]) -> dict:
    errors, heights, n_lines, texts = [], [], [], []
    for r in rows:
        boxes, h, w = r["boxes"], r["height"], r["width"]
        n_lines.append(len(boxes))
        if len(boxes) < MIN_LINES:
            errors.append(f"{r['image']}: only {len(boxes)} lines")
        scale = 1024 / max(h, w)
        polys = []
        for b in boxes:
            p = np.asarray(b["polygon"], np.float64)
            texts.append(b["text"])
            if p.shape != (4, 2) or (p[:, 0] < 0).any() or (p[:, 0] > w).any() or (p[:, 1] < 0).any() or (p[:, 1] > h).any():
                errors.append(f"{r['image']}: polygon outside image {b['text'][:20]!r}")
            if not b["text"].strip():
                errors.append(f"{r['image']}: empty text")
            x1, y1, x2, y2 = b["bbox"]
            if not (x2 > x1 and y2 > y1):
                errors.append(f"{r['image']}: degenerate bbox")
            heights.append(min(np.linalg.norm(p[1] - p[0]), np.linalg.norm(p[3] - p[0])) * scale)
            polys.append(b["polygon"])
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polygon_iou(polys[i], polys[j]) > MAX_IOU:
                    errors.append(f"{r['image']}: lines {i},{j} overlap")
                    break
    hs = np.asarray(heights) if heights else np.zeros(1)
    return {"n_images": len(rows), "n_lines_total": int(sum(n_lines)), "lines_per_image": {"min": int(min(n_lines, default=0)), "median": float(np.median(n_lines)) if n_lines else 0, "max": int(max(n_lines, default=0))},
            "text_height_px_at_1024": {"p5": float(np.percentile(hs, 5)), "p50": float(np.median(hs)), "p95": float(np.percentile(hs, 95))},
            "labels": dict(sorted(Counter(b["label"] for r in rows for b in r["boxes"]).items())),
            "errors": errors[:50], "n_errors": len(errors), "texts": texts}


def check_lines(lines_dir: Path) -> dict:
    gt = lines_dir / "gt.txt"
    rows = [l.rstrip("\n").split("\t", 1) for l in open(gt, encoding="utf-8")] if gt.exists() else []
    errors, texts, widths = [], [], []
    for path, text in rows:
        texts.append(text)
        with Image.open(lines_dir / path) as im:
            w, h = im.size
        widths.append(w)
        if h != LINE_H or w > LINE_MAX_W or w < 1:
            errors.append(f"{path}: size {w}x{h}")
        if len(text) > LINE_MAX_LEN:
            errors.append(f"{path}: label too long")
    non_blank = sum(bool(t.strip()) for t in texts)
    return {"n": len(rows), "width": {"p5": float(np.percentile(widths, 5)) if widths else 0, "p50": float(np.median(widths)) if widths else 0, "p95": float(np.percentile(widths, 95)) if widths else 0},
            "label_len_p50": float(np.median([len(t) for t in texts])) if texts else 0,
            "non_blank_ratio": non_blank / len(texts) if texts else 0, "errors": errors[:50], "n_errors": len(errors), "texts": texts}


def bench_line_renderer(corpus_lines, vocab, fonts, n=300) -> float:
    c, r, rng = ReceiptCorpus(corpus_lines, vocab, seed=0), LineRenderer(fonts), random.Random(0)
    t0 = time.perf_counter()
    for _ in range(n):
        r.render(c.line(rng), rng)
    return (time.perf_counter() - t0) / n * 1000


def preview(split_dir: Path, rows: list[dict], out: Path, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    files = []
    for g in range(0, len(picked), 6):
        tiles = []
        for r in picked[g : g + 6]:
            img = cv2.imread(str(split_dir / r["image"]))
            for b in r["boxes"]:
                cv2.polylines(img, [np.asarray(b["polygon"], np.int32)], True, (0, 0, 255), 2)
            h, w = img.shape[:2]
            s = 640 / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)))
            cv2.putText(img, r["image"], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(cv2.copyMakeBorder(img, 0, 640 - img.shape[0], 0, 640 - img.shape[1], cv2.BORDER_CONSTANT, value=(128, 128, 128)))
        while len(tiles) % 3:
            tiles.append(np.full((640, 640, 3), 128, np.uint8))
        grid = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
        p = out / "preview" / f"receipts_{g // 6:02d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        files.append(p.as_posix())
    return files


def to_md(rep: dict) -> str:
    L = ["# Synthetic OCR data report", "", f"splits.json sha256: `{rep['splits_sha256']}`", f"corpus sha256: `{rep['corpus_sha256']}`", "", "## Checks", ""]
    L += [f"- {'PASS' if v else 'FAIL'} — {k}" for k, v in rep["checks"].items()]
    L += ["", "## Receipts", ""]
    for split, s in rep["receipts"].items():
        th = s["text_height_px_at_1024"]
        L.append(f"- **{split}**: {s['n_images']} images, {s['n_lines_total']} lines (per image min/med/max {s['lines_per_image']['min']}/{s['lines_per_image']['median']:.0f}/{s['lines_per_image']['max']}); "
                 f"text height@1024 p5/p50/p95 = {th['p5']:.1f}/{th['p50']:.1f}/{th['p95']:.1f} px; labels {s['labels']}; errors {s['n_errors']}")
        L += [f"  - {e}" for e in s["errors"][:10]]
    s = rep["lines"]
    L += ["", "## Lines (val)", "", f"- n={s['n']}, width p5/p50/p95 = {s['width']['p5']:.0f}/{s['width']['p50']:.0f}/{s['width']['p95']:.0f}, label len p50 = {s['label_len_p50']:.0f}, non-blank {s['non_blank_ratio']:.1%}, errors {s['n_errors']}",
          f"- LineRenderer: {rep['line_render_ms']:.2f} ms/line (budget {LINE_MS_BUDGET} ms)"]
    p = rep["provenance"]
    L += ["", "## Charset / provenance", "", f"- chars outside vocab: {rep['bad_chars']!r}",
          f"- corpus lines NOT sourced from train: {len(rep['leaked'])}"]
    L += [f"  - {x}" for x in rep["leaked"][:10]]
    L += [f"- info: {p['heldout_in_train']}/{p['heldout_texts']} val/test box texts ({p['heldout_overlap_ratio']:.1%}) also occur verbatim in train "
          "(same store chains across splits). Read val/test CER / Field EM with this in mind."]
    if rep.get("preview"):
        L += ["", "## Preview (AC-4, owner review)", ""] + [f"- `{p}`" for p in rep["preview"]]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synth_ocr")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--ann", default="data/processed/annotations.jsonl")
    ap.add_argument("--fonts", default="data/fonts")
    ap.add_argument("--out", default="results/data/synth_ocr")
    ap.add_argument("--preview", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    root, out = Path(args.root), Path(args.out)
    splits = load_splits(args.splits)
    vocab = load_vocab_set(root / "corpus" / "vocab.txt")
    corpus_lines = [l.rstrip("\n") for l in open(root / "corpus" / "receipt_lines.txt", encoding="utf-8") if l.strip()]
    fonts = sorted(str(f) for f in Path(args.fonts).glob("*.ttf"))

    receipts, all_texts, generated = {}, list(corpus_lines), []
    for split in ("train", "val"):
        idx = root / "receipts" / split / "index.jsonl"
        if idx.exists():
            rows = _load_index(idx)
            receipts[split] = check_receipts(root / "receipts" / split, rows)
            generated += receipts[split].pop("texts")
    lines = check_lines(root / "lines" / "val")
    generated += lines.pop("texts")
    all_texts += generated

    bad_chars = check_charset(all_texts, vocab)
    train_texts, heldout_texts = _split_texts(Path(args.ann), splits, vocab)
    leaked, provenance = check_provenance(corpus_lines, train_texts, heldout_texts)
    heldout_hits = check_heldout_generated(generated, train_texts, heldout_texts)
    ms = bench_line_renderer(corpus_lines, vocab, fonts)
    heights_ok = all(HEIGHT_RANGE[0] <= s["text_height_px_at_1024"]["p50"] <= HEIGHT_RANGE[1] for s in receipts.values())

    rep = {
        "splits_sha256": _sha(Path(args.splits)), "corpus_sha256": _sha(root / "corpus" / "receipt_lines.txt"),
        "receipts": receipts, "lines": lines, "bad_chars": "".join(bad_chars), "leaked": leaked, "provenance": provenance, "heldout_generated": heldout_hits, "line_render_ms": ms,
        "checks": {
            "charset": not bad_chars,
            "no_leak": not leaked,
            "no_heldout_generated": not heldout_hits,
            "polygons_valid": bool(receipts) and all(s["n_errors"] == 0 for s in receipts.values()),
            "text_height": bool(receipts) and heights_ok,
            "lines_valid": lines["n"] > 0 and lines["n_errors"] == 0 and lines["non_blank_ratio"] >= 0.95,
            "line_render_speed": ms <= LINE_MS_BUDGET,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    if args.preview > 0 and "train" in receipts:
        rep["preview"] = preview(root / "receipts" / "train", _load_index(root / "receipts" / "train" / "index.jsonl"), out, args.preview, args.seed)
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(to_md(rep), encoding="utf-8")
    print(to_md(rep))
    return 0 if all(rep["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
