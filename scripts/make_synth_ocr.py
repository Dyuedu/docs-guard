"""Generate synthetic OCR data (intent 002).

    # 1. corpus from MC-OCR *train* split only
    python scripts/make_synth_ocr.py corpus --ann data/processed/annotations.jsonl \
        --splits data/processed/splits.json --vocab configs/vocab_vi.txt --out data/synth_ocr/corpus

    # 2. full-page receipts for DBNet (polygons for every line)
    python scripts/make_synth_ocr.py receipts --corpus data/synth_ocr/corpus --fonts data/fonts \
        --out data/synth_ocr/receipts --n-train 20000 --n-val 1000 --workers 8 --seed 42

    # 3. a fixed validation set of text lines for CRNN (train lines are generated on the fly)
    python scripts/make_synth_ocr.py lines --corpus data/synth_ocr/corpus --fonts data/fonts \
        --out data/synth_ocr/lines/val --n 5000 --seed 42

Validate with scripts/synth_ocr_report.py.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from docguard.data.annotations import read_jsonl
from docguard.data.splits import load_splits
from docguard.data.synth_receipt import DEFAULT_CFG, render_receipt
from docguard.data.synth_text import LineRenderer, ReceiptCorpus, filter_charset, heldout_only, load_vocab_set

_W: dict = {}  # per-process worker state


def load_corpus_dir(corpus_dir: str | Path) -> tuple[list[str], set[str], set[str]]:
    d = Path(corpus_dir)
    texts = [l.rstrip("\n") for l in open(d / "receipt_lines.txt", encoding="utf-8") if l.strip()]
    vocab = load_vocab_set(d / "vocab.txt")
    bl = d / "heldout_blocklist.txt"
    blocklist = {l.rstrip("\n") for l in open(bl, encoding="utf-8") if l.strip()} if bl.exists() else set()
    return texts, vocab, blocklist


def resolve_fonts(path: str | Path) -> list[str]:
    p = Path(path)
    fonts = [str(p)] if p.is_file() else sorted(str(f) for f in p.glob("*.ttf"))
    if not fonts:
        raise FileNotFoundError(f"no .ttf under {p}")
    return fonts


# --- corpus ---------------------------------------------------------------------------------------

def cmd_corpus(args) -> None:
    splits = load_splits(args.splits)
    vocab = load_vocab_set(args.vocab)
    samples = read_jsonl(args.ann)
    texts, words, held = [], set(), []
    for s in samples:
        if s.image not in splits["train"]:
            # val/test texts never enter the corpus (AC-2); they only feed the blocklist (clause 2)
            if s.image in splits["val"] | splits["test"]:
                held += [" ".join(filter_charset(b.text, vocab).split()) for b in s.boxes]
            continue
        for b in s.boxes:
            t = " ".join(filter_charset(b.text, vocab).split())
            if t:
                texts.append(t)
                words.update(w for w in t.split() if len(w) > 1)
    blocklist = sorted(heldout_only(texts, held))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout_blocklist.txt").write_text("\n".join(blocklist) + "\n", encoding="utf-8")
    (out / "receipt_lines.txt").write_text("\n".join(texts) + "\n", encoding="utf-8")
    (out / "words.txt").write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    (out / "vocab.txt").write_text(Path(args.vocab).read_text(encoding="utf-8"), encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"ann": args.ann, "splits": args.splits, "n_train_images": len(splits["train"]),
                                               "n_lines": len(texts), "n_words": len(words), "n_blocklist": len(blocklist)}, indent=1), encoding="utf-8")
    print(f"corpus: {len(texts)} lines, {len(words)} words from {len(splits['train'])} train receipts, {len(blocklist)} held-out-only texts blocked -> {out}")


# --- receipts -------------------------------------------------------------------------------------

def _init_receipts(corpus_dir, fonts, out_dir, cfg):
    texts, vocab, blocklist = load_corpus_dir(corpus_dir)
    _W["corpus"] = ReceiptCorpus(texts, vocab, blocklist=blocklist)
    _W["fonts"], _W["out"], _W["cfg"] = fonts, Path(out_dir), cfg


def _one_receipt(task):
    idx, seed = task
    rng = random.Random(seed)
    lines = _W["corpus"].receipt(rng)
    img, boxes = render_receipt(lines, _W["fonts"], rng, _W["cfg"])
    name = f"images/{idx:06d}.jpg"
    q = rng.randint(60, 95)  # SynthDoG quality [50, 95]; MC-OCR photos are phone JPEGs
    cv2.imwrite(str(_W["out"] / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return {"image": name, "height": int(img.shape[0]), "width": int(img.shape[1]), "jpeg_q": q, "seed": seed, "boxes": boxes}


def generate_receipts(corpus_dir, fonts, out_dir, n: int, seed: int, workers: int, cfg=None, log=print) -> list[dict]:
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    tasks = [(i, seed * 1_000_003 + i) for i in range(n)]
    t0 = time.perf_counter()
    records = []
    if workers <= 1:
        _init_receipts(corpus_dir, fonts, out, cfg)
        for k, t in enumerate(tasks):
            records.append(_one_receipt(t))
            if (k + 1) % 500 == 0:
                log(f"  {k + 1}/{n}")
    else:
        with ProcessPoolExecutor(workers, initializer=_init_receipts, initargs=(corpus_dir, fonts, out, cfg)) as ex:
            for k, rec in enumerate(ex.map(_one_receipt, tasks, chunksize=16)):
                records.append(rec)
                if (k + 1) % 500 == 0:
                    log(f"  {k + 1}/{n}")
    records.sort(key=lambda r: r["image"])
    with open(out / "index.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    log(f"{out}: {n} receipts in {time.perf_counter() - t0:.0f}s")
    return records


def cmd_receipts(args) -> None:
    fonts = resolve_fonts(args.fonts)
    for split, n, s in (("train", args.n_train, args.seed), ("val", args.n_val, args.seed + 1)):
        if n > 0:
            generate_receipts(args.corpus, fonts, Path(args.out) / split, n, s, args.workers)


# --- lines ----------------------------------------------------------------------------------------

def generate_lines(corpus_dir, fonts, out_dir, n: int, seed: int, log=print) -> list[tuple[str, str]]:
    texts, vocab, blocklist = load_corpus_dir(corpus_dir)
    corpus, renderer = ReceiptCorpus(texts, vocab, blocklist=blocklist), LineRenderer(fonts)
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    t0 = time.perf_counter()
    for i in range(n):
        text = corpus.line(rng)[:40]
        while corpus.blocked(text):  # truncation could still recreate a blocked text
            text = corpus.line(rng)[:40]
        img = renderer.render(text, rng)
        name = f"images/{i:06d}.jpg"
        cv2.imwrite(str(out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(70, 95)])
        rows.append((name, text))
    with open(out / "gt.txt", "w", encoding="utf-8") as f:
        f.writelines(f"{p}\t{t}\n" for p, t in rows)
    log(f"{out}: {n} lines in {time.perf_counter() - t0:.1f}s")
    return rows


def cmd_lines(args) -> None:
    generate_lines(args.corpus, resolve_fonts(args.fonts), args.out, args.n, args.seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--ann", required=True); c.add_argument("--splits", required=True)
    c.add_argument("--vocab", default="configs/vocab_vi.txt"); c.add_argument("--out", default="data/synth_ocr/corpus")
    c.set_defaults(fn=cmd_corpus)
    r = sub.add_parser("receipts")
    r.add_argument("--corpus", default="data/synth_ocr/corpus"); r.add_argument("--fonts", default="data/fonts")
    r.add_argument("--out", default="data/synth_ocr/receipts"); r.add_argument("--n-train", type=int, default=20000)
    r.add_argument("--n-val", type=int, default=1000); r.add_argument("--workers", type=int, default=8)
    r.add_argument("--seed", type=int, default=42)
    r.set_defaults(fn=cmd_receipts)
    l = sub.add_parser("lines")
    l.add_argument("--corpus", default="data/synth_ocr/corpus"); l.add_argument("--fonts", default="data/fonts")
    l.add_argument("--out", default="data/synth_ocr/lines/val"); l.add_argument("--n", type=int, default=5000)
    l.add_argument("--seed", type=int, default=42)
    l.set_defaults(fn=cmd_lines)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args.fn(args)


if __name__ == "__main__":
    main()
