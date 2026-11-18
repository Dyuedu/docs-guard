"""Spec 003 AC-8 (a): 2-iteration CPU smoke run writes a checkpoint and an MLflow run."""

import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("mlflow")
pytest.importorskip("pyclipper")

from tests.test_det_labels import _write_fixture


def test_smoke_two_iters(tmp_path, monkeypatch):
    from docguard.train.loop import train_det

    data = tmp_path / "data"
    data.mkdir()
    jp = _write_fixture(data, n=2)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")  # scoped: setup_mlflow uses setdefault
    cfg = {
        "model": "dbnet", "seed": 0, "pretrained": False, "run_name": "smoke",
        "data": {"train": [jp], "val": jp, "image_root": None, "input_size": 960, "crop_size": 64,
                 "scale_range": [0.7, 1.3], "rotate_deg": 5, "min_text_size": 8,
                 "shrink_ratio": 0.4, "thresh_min": 0.3, "thresh_max": 0.7},
        "train": {"iters": 2, "batch_size": 2, "lr": 0.001, "weight_decay": 0.0001, "amp": True,
                  "num_workers": 0, "warmup_iters": 1, "eval_every": 2, "log_every": 1,
                  "early_stop_patience": 5, "ohem_ratio": 3.0,
                  "loss_weights": {"prob": 5.0, "binary": 1.0, "thresh": 10.0}, "init": None},
        "out_dir": str(tmp_path / "out"), "mlflow_experiment": "text_det_smoke",
    }
    res = train_det(cfg, device="cpu")
    assert res["iters"] == 2 and res["run_id"]
    ck = torch.load(tmp_path / "out" / "smoke" / "best.pt", map_location="cpu", weights_only=False)
    assert set(ck) == {"model", "config", "iter", "hmean", "mlflow_run_id"}
    assert ck["iter"] == 2 and ck["mlflow_run_id"] == res["run_id"]
    assert (tmp_path / "out" / "smoke" / "last.pt").exists()

    import mlflow

    run = mlflow.get_run(res["run_id"])
    assert math.isfinite(run.data.metrics["loss"])
    assert "val_hmean" in run.data.metrics


def test_eval_det_script_logs_to_default_mlruns(tmp_path):
    # Verifier finding: scripts/eval_det.py logs to the checkpoint's MLflow run, but without the
    # file-store setup of loop.py (plan D4) MLflow 3.x cannot find ./mlruns and the script crashes.
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "d").mkdir()
    jp = _write_fixture(tmp_path / "d", n=1)
    env = {k: v for k, v in os.environ.items() if not k.startswith("MLFLOW_")}
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), str(repo)])
    cfg = {
        "model": "dbnet", "seed": 0, "pretrained": False, "run_name": "r",
        "data": {"train": [jp], "val": jp, "image_root": None, "input_size": 960, "crop_size": 64,
                 "scale_range": [1.0, 1.0], "rotate_deg": 0, "min_text_size": 8,
                 "shrink_ratio": 0.4, "thresh_min": 0.3, "thresh_max": 0.7},
        "train": {"iters": 1, "batch_size": 1, "lr": 0.001, "amp": False, "num_workers": 0,
                  "warmup_iters": 0, "eval_every": 1, "log_every": 1},
        "out_dir": "out", "mlflow_experiment": "t",
    }
    train = f"from docguard.train.loop import train_det; train_det({cfg!r}, device='cpu')"
    subprocess.run([sys.executable, "-c", train], cwd=tmp_path, env=env, check=True, capture_output=True)
    out = tmp_path / "res.json"
    r = subprocess.run([sys.executable, str(repo / "scripts/eval_det.py"), "--ckpt", "out/r/best.pt",
                        "--data", "synth_val", "--out", str(out), "--device", "cpu"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(out.read_text(encoding="utf-8"))["mlflow_run_id"]


def test_imgs_per_s_excludes_eval_time(tmp_path, monkeypatch):
    # Bug seen on the rented GPU: imgs_per_s logged right after an eval included the eval time
    # (14 img/s vs 73 img/s), which would wrongly trip the spec's "stop if < 37 img/s" rule.
    import time

    import mlflow

    import docguard.train.loop as loop

    def slow_eval(*a, **k):
        time.sleep(3.0)
        return {"precision": 0.0, "recall": 0.0, "hmean": 0.0, "mean_iou": 0.0}

    monkeypatch.setattr(loop, "evaluate_det", slow_eval)
    (tmp_path / "d").mkdir()
    jp = _write_fixture(tmp_path / "d", n=2)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    cfg = {
        "model": "dbnet", "seed": 0, "pretrained": False, "run_name": "ips",
        "data": {"train": [jp], "val": jp, "image_root": None, "input_size": 960, "crop_size": 64,
                 "scale_range": [1.0, 1.0], "rotate_deg": 0, "min_text_size": 8,
                 "shrink_ratio": 0.4, "thresh_min": 0.3, "thresh_max": 0.7},
        "train": {"iters": 2, "batch_size": 2, "lr": 0.001, "amp": False, "num_workers": 0,
                  "warmup_iters": 0, "eval_every": 1, "log_every": 2},
        "out_dir": str(tmp_path / "out"), "mlflow_experiment": "ips",
    }
    res = loop.train_det(cfg, device="cpu")
    ips = mlflow.get_run(res["run_id"]).data.metrics["imgs_per_s"]
    # Window = iter 2 only (2 images, well under 1 s on CPU) once the eval at iter 1 is excluded;
    # with the 3 s eval included it is at most 4 images / 3 s = 1.33 img/s.
    assert ips > 2.0, ips
