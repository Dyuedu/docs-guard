from pathlib import Path

import pytest

from docguard.config import CONFIG_DIR, load_config, parse_overrides, pipeline_config


def test_parse_overrides_nested_and_typed():
    assert parse_overrides(["train.lr=1e-3", "train.amp=false", "data.root=x"]) == {
        "train": {"lr": 1e-3, "amp": False},
        "data": {"root": "x"},
    }


def test_parse_overrides_rejects_missing_equals():
    with pytest.raises(ValueError):
        parse_overrides(["train.lr"])


@pytest.mark.parametrize("name", ["text_det", "text_rec", "tamper_seg"])
def test_train_configs_load_and_override(name):
    cfg = load_config(CONFIG_DIR / "train" / f"{name}.yaml", ["train.epochs=1"])
    assert cfg["train"]["epochs"] == 1
    assert cfg["seed"] == 42


def test_train_configs_match_serving_shapes():
    pipe = pipeline_config()
    det = load_config(CONFIG_DIR / "train" / "text_det.yaml")
    rec = load_config(CONFIG_DIR / "train" / "text_rec.yaml")
    seg = load_config(CONFIG_DIR / "train" / "tamper_seg.yaml")
    assert det["data"]["input_size"] == pipe["text_det"]["input_size"]
    assert (rec["data"]["height"], rec["data"]["width"]) == (pipe["text_rec"]["height"], pipe["text_rec"]["width"])
    assert seg["data"]["patch_size"] == pipe["tampering"]["patch_size"]
    assert Path(CONFIG_DIR.parent / rec["data"]["vocab_file"]).exists()
