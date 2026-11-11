"""Validate the synthetic tampering dataset and write a report (intent 001, AC-1..AC-4, AC-6, AC-7).

    python scripts/tamper_report.py --root data/tamper --splits data/processed/splits.json \
        --out results/data/tamper --preview 50

Writes report.json, report.md and preview/grid_XX.jpg (train/val only, stratified by op).
Exit code 1 if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

from docguard.data.splits import SPLITS, load_splits
from docguard.data.tamper_stats import TAMPER_OPS, leak_check, load_index, mask_stats, preview_grid, summarize

PER_GRID = 10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pick_preview(index: dict[str, list[dict]], n: int, seed: int) -> list[tuple[str, dict]]:
    """n tampered samples from train+val (never test), round-robin over ops."""
    rng = random.Random(seed)
    pools = {op: [(s, r) for s in ("train", "val") for r in index.get(s, []) if r["op"] == op] for op in TAMPER_OPS}
    for p in pools.values():
        rng.shuffle(p)
    picked: list[tuple[str, dict]] = []
    while len(picked) < n and any(pools.values()):
        for op in TAMPER_OPS:
            if pools[op] and len(picked) < n:
                picked.append(pools[op].pop())
    return picked


def to_markdown(rep: dict) -> str:
    lines = ["# Tamper dataset report", "", f"splits.json sha256: `{rep['splits_sha256']}`", "", "## Checks", ""]
    lines += [f"- {'PASS' if ok else 'FAIL'} — {name}" for name, ok in rep["checks"].items()]
    lines += ["", "## Per split", "", "| split | images | authentic | tampered | sources | copy_move | splice | inpaint | median area | p95 area |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for split, s in rep["splits"].items():
        a = s["area_ratio"]["all"]
        ratios = " | ".join(f"{s['op_counts'][op]} ({s['op_ratio'][op]:.0%})" for op in TAMPER_OPS)
        lines.append(f"| {split} | {s['n_images']} | {s['n_authentic']} | {s['n_tampered']} | {s['n_sources']} | {ratios} | "
                     f"{a.get('p50', 0):.5f} | {a.get('p95', 0):.5f} |")
    lines += ["", "## Mask area ratio by op (all splits shown separately)", ""]
    for split, s in rep["splits"].items():
        for op in TAMPER_OPS:
            p = s["area_ratio"][op]
            if p:
                lines.append(f"- {split}/{op}: p5={p['p5']:.5f} p50={p['p50']:.5f} p95={p['p95']:.5f} max={p['max']:.5f} (n={p['n']})")
    lines += ["", "## Fonts (inpaint)", ""]
    lines += [f"- {split}: {s['fonts']}" for split, s in rep["splits"].items()]
    lines += ["", f"## Outliers (area ratio > {rep['config']['outlier_area_ratio']}) — review manually: {len(rep['outliers'])}", ""]
    lines += [f"- {o['split']}/{o['name']} ({o['op']}): {o['area_ratio']:.4f}" for o in rep["outliers"][:50]]
    lines += ["", f"## Invalid masks: {len(rep['invalid'])}", ""]
    lines += [f"- {i['split']}/{i['name']}: {i['error']}" for i in rep["invalid"][:50]]
    lines += ["", f"## Leak errors: {len(rep['leak_errors'])}", ""]
    lines += [f"- {e}" for e in rep["leak_errors"][:50]]
    if rep.get("preview"):
        lines += ["", "## Preview (AC-7)", "", "Each row: full image with red bbox | zoomed crop | crop with mask overlay.", ""]
        lines += [f"- `{p}`" for p in rep["preview"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tamper")
    ap.add_argument("--splits", default="data/processed/splits.json")
    ap.add_argument("--out", default="results/data/tamper")
    ap.add_argument("--preview", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to a legacy codepage

    root, out = Path(args.root), Path(args.out)
    splits = load_splits(args.splits)
    index = {s: load_index(root / s) for s in SPLITS}
    stats = {s: mask_stats(root / s, index[s]) for s in SPLITS}
    rep = summarize(index, stats, leak_check(index, splits))
    rep["splits_sha256"] = _sha256(Path(args.splits))

    out.mkdir(parents=True, exist_ok=True)
    if args.preview > 0:
        picked = pick_preview(index, args.preview, args.seed)
        rep["preview"] = []
        for split in ("train", "val"):
            rows = [r for s, r in picked if s == split]
            for g in range(0, len(rows), PER_GRID):
                path = out / "preview" / f"{split}_{g // PER_GRID:02d}.jpg"
                preview_grid(root / split, rows[g : g + PER_GRID], path)
                rep["preview"].append(path.as_posix())
        rep["preview_rows"] = [{"split": s, "name": r["name"], "op": r["op"]} for s, r in picked]

    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep).split("## Per split")[0].strip())
    print(f"report -> {out / 'report.md'}")
    return 0 if all(rep["checks"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
