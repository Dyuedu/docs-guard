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
