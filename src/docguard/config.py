"""Shared YAML config loading for pipeline, experiments and training configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Turn ["train.lr=1e-3", "data.root=x"] into a nested dict; values parsed as YAML scalars."""
    out: dict[str, Any] = {}
    for item in items:
        key, _, raw = item.partition("=")
        if not key or not _:
            raise ValueError(f"override must be key=value, got {item!r}")
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _scalar(raw)
    return out


def _scalar(raw: str) -> Any:
    value = yaml.safe_load(raw)
    if isinstance(value, str):  # YAML 1.1 reads "1e-3" as a string; accept it as a float
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    cfg = load_yaml(path)
    if overrides:
        _deep_update(cfg, parse_overrides(overrides))
    return cfg


def pipeline_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "pipeline.yaml")


def experiments_config() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "experiments.yaml")
