"""Train a model from a YAML config (intent 003: DBNet text detection).

    python scripts/train.py --config configs/train/text_det.yaml [--set train.iters=200 ...]
"""

from __future__ import annotations

import argparse
import json

from docguard.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override, e.g. train.iters=200")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if cfg.get("model") != "dbnet":
        raise SystemExit(f"unsupported model {cfg.get('model')!r}; only dbnet is implemented")
    from docguard.train.loop import train_det

    print(json.dumps(train_det(cfg, device=args.device), indent=2))


if __name__ == "__main__":
    main()
