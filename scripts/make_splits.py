"""Create the canonical split file from normalized MC-OCR annotations.

    python scripts/make_splits.py --ann data/processed/annotations.jsonl --out data/processed/splits.json
"""

from __future__ import annotations

import argparse

from docguard.data.annotations import read_jsonl
from docguard.data.splits import make_splits, save_splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", default="data/processed/splits.json")
    ap.add_argument("--split", default="0.8,0.1,0.1")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fr = tuple(float(x) for x in args.split.split(","))
    splits = make_splits([s.image for s in read_jsonl(args.ann)], fr, args.seed)
    save_splits(splits, args.out)
    print({k: len(splits[k]) for k in ("train", "val", "test")}, "->", args.out)


if __name__ == "__main__":
    main()
