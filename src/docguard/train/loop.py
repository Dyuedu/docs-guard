"""Iteration-based DBNet training loop (spec 003): Adam + warmup/poly LR, AMP, periodic
evaluate_det on the val set, best/last checkpoints, early stopping, MLflow logging.

Loop structure cross-checked against WenmuZhou/DBNet.pytorch (trainer) and PaddleOCR
tools/program.py; written fresh for this repo.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from docguard.config import pipeline_config
from docguard.models.dbnet import DBNet
from docguard.train.det_dataset import DetDataset, EpochOffsetSampler
from docguard.train.det_loss import DBLoss
from docguard.train.eval_det import evaluate_det


def lr_at(it: int, base_lr: float, iters: int, warmup: int, power: float = 0.9) -> float:
    if warmup and it < warmup:
        return base_lr * (it + 1) / warmup
    return base_lr * (1 - it / iters) ** power


def make_predict_prob(model: torch.nn.Module, device: str):
    @torch.no_grad()
    def predict(inp: np.ndarray) -> np.ndarray:
        return model(torch.from_numpy(inp).to(device))[0, 0].float().cpu().numpy()

    return predict


def _flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(_flatten(v, key + ".") if isinstance(v, dict) else {key: str(v)})
    return out


def _check_shrink_ratio(cfg: dict, pipe: dict) -> None:
    a, b = cfg["data"]["shrink_ratio"], pipe["text_det"]["shrink_ratio"]
    if a != b:
        raise ValueError(f"data.shrink_ratio {a} != pipeline.yaml text_det.shrink_ratio {b}")


def train_det(cfg: dict, device: str | None = None) -> dict:
    """Train per cfg (configs/train/text_det.yaml layout). Returns run id, best H-mean, ckpt dir."""
    import mlflow

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d, t = cfg["data"], cfg["train"]
    pipe = pipeline_config()
    _check_shrink_ratio(cfg, pipe)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    train_ds = DetDataset(d["train"], d.get("image_root"), True, d, seed)
    val_paths = d["val"] if isinstance(d["val"], list) else [d["val"]]
    val_ds = DetDataset(val_paths, d.get("image_root"), False, d, seed)
    workers = int(t.get("num_workers", 0))
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=t["batch_size"], sampler=EpochOffsetSampler(len(train_ds), seed),
        num_workers=workers, pin_memory=device == "cuda", persistent_workers=workers > 0, drop_last=True,
    )

    model = DBNet(pretrained=bool(cfg.get("pretrained", True))).to(device)
    if t.get("init"):
        model.load_state_dict(torch.load(t["init"], map_location=device)["model"])
    lw = t.get("loss_weights", {})
    criterion = DBLoss(alpha=lw.get("prob", 5.0), beta=lw.get("thresh", 10.0), ohem_ratio=t.get("ohem_ratio", 3.0))
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t.get("weight_decay", 0.0))
    use_amp = bool(t.get("amp", False)) and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    iters, warmup = int(t["iters"]), int(t.get("warmup_iters", 1000))
    eval_every, patience = int(t.get("eval_every", iters)), int(t.get("early_stop_patience", 0))
    log_every = int(t.get("log_every", 50))
    run_name = cfg.get("run_name") or time.strftime("%Y%m%d-%H%M%S")
    ckpt_dir = Path(cfg["out_dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
    if "://" not in uri or uri.startswith("file:"):
        # Spec 003: runs live in a plain mlruns/ folder the owner copies back from the rented GPU.
        # MLflow >= 3.x refuses the file store unless this opt-out is set.
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg.get("mlflow_experiment", "text_det"))
    best, stale, it = -1.0, 0, 0
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        mlflow.log_params(_flatten(cfg))
        mlflow.log_param("num_params", sum(p.numel() for p in model.parameters()))
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        def save(name: str, hmean: float) -> Path:
            p = ckpt_dir / name
            torch.save({"model": model.state_dict(), "config": cfg, "iter": it, "hmean": hmean,
                        "mlflow_run_id": run_id}, p)
            return p

        def evaluate() -> float:
            model.eval()
            m = evaluate_det(make_predict_prob(model, device), val_ds, pipe)
            model.train()
            mlflow.log_metrics({f"val_{k}": m[k] for k in ("precision", "recall", "hmean", "mean_iou")}, step=it)
            return m["hmean"]

        model.train()
        t0, seen = time.time(), 0
        for batch in loader:
            if it >= iters:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(it, t["lr"], iters, warmup)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device, dtype=torch.float16, enabled=use_amp):
                out = model(batch["image"])
            losses = criterion(out, batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(losses["loss"]).backward()
            scaler.step(opt)
            scaler.update()
            it += 1
            seen += batch["image"].shape[0]
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"non-finite loss at iter {it}")
            if it % log_every == 0 or it == iters:
                dt = time.time() - t0
                mlflow.log_metrics({k: float(v) for k, v in losses.items()} |
                                   {"lr": opt.param_groups[0]["lr"], "imgs_per_s": seen / max(dt, 1e-9)}, step=it)
                t0, seen = time.time(), 0
            if it % eval_every == 0 or it == iters:
                h = evaluate()
                save("last.pt", h)
                if h > best:
                    best, stale = h, 0
                    save("best.pt", h)
                else:
                    stale += 1
                    if patience and stale >= patience:
                        break
        if device == "cuda":
            mlflow.log_metric("max_vram_gb", torch.cuda.max_memory_allocated() / 2**30)
        mlflow.log_metric("best_val_hmean", best)
        mlflow.log_metric("best_pt_mb", (ckpt_dir / "best.pt").stat().st_size / 2**20)
        mlflow.log_artifact(str(ckpt_dir / "best.pt"))
    return {"run_id": run_id, "best_hmean": best, "ckpt_dir": str(ckpt_dir), "iters": it}
