"""Evaluate a DBNet checkpoint through the serving path (spec 003 AC-10).

    python scripts/eval_det.py --ckpt <best.pt> --data synth_val|mcocr_val \
        --out results/text_det/<name>.json [--mlflow-run-id <id>]

synth_val : data.val of the checkpoint config (H-mean over every box).
mcocr_val : data/processed/annotations.jsonl filtered by splits.json["val"]; recall over the 4 KIE
            fields only (SELLER, ADDRESS, TIMESTAMP, TOTAL_COST).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MCOCR_LABELS = {"SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"}
MCOCR_ANN = "data/processed/annotations.jsonl"
MCOCR_SPLITS = "data/processed/splits.json"
MCOCR_IMAGE_DIRS = ("data/raw/mcocr/train_images", "data/raw/mcocr/val_images")


def mcocr_val_samples(ann: str = MCOCR_ANN, splits: str = MCOCR_SPLITS):
    from docguard.train.det_dataset import load_rgb

    val = set(json.loads(Path(splits).read_text(encoding="utf-8"))["val"])
    with open(ann, encoding="utf-8") as f:
        recs = [r for r in map(json.loads, filter(str.strip, f)) if r["image"] in val]
    for r in recs:
        path = next((Path(d) / r["image"] for d in MCOCR_IMAGE_DIRS if (Path(d) / r["image"]).exists()), None)
        if path is None:
            raise FileNotFoundError(r["image"])
        yield {"image": load_rgb(str(path)), "boxes": [b["bbox"] for b in r["boxes"]],
               "labels": [b.get("label") for b in r["boxes"]], "name": r["image"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, choices=["synth_val", "mcocr_val"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlflow-run-id", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    from docguard.config import pipeline_config
    from docguard.models.dbnet import DBNet
    from docguard.train.det_dataset import DetDataset
    from docguard.train.eval_det import evaluate_det
    from docguard.train.loop import make_predict_prob

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = DBNet(pretrained=False).to(device).eval()
    model.load_state_dict(ck["model"])
    cfg = ck["config"]

    if args.data == "synth_val":
        d = cfg["data"]
        val = d["val"] if isinstance(d["val"], list) else [d["val"]]
        samples, labels = DetDataset(val, d.get("image_root"), False, d, 0), None
    else:
        samples, labels = mcocr_val_samples(), MCOCR_LABELS

    m = evaluate_det(make_predict_prob(model, device), samples, pipeline_config(), labels=labels)
    m.update({"data": args.data, "ckpt": args.ckpt, "ckpt_iter": ck.get("iter"),
              "mlflow_run_id": args.mlflow_run_id or ck.get("mlflow_run_id")})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: m[k] for k in ("precision", "recall", "hmean", "mean_iou", "n_gt")}, indent=2))

    if m["mlflow_run_id"]:
        import mlflow

        from docguard.train.loop import setup_mlflow

        setup_mlflow()
        with mlflow.start_run(run_id=m["mlflow_run_id"]):
            mlflow.log_metrics({f"{args.data}_{k}": m[k] for k in ("precision", "recall", "hmean", "mean_iou")})
            mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
