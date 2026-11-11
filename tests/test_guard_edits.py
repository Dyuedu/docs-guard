"""Tests for the AI-native SDLC edit guard (.claude/hooks/guard_edits.py)."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_edits.py"
_spec = importlib.util.spec_from_file_location("guard_edits", HOOK)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _ev(root, rel, **extra):
    return {"tool_input": {"file_path": str(Path(root) / rel), **extra}}


@pytest.mark.parametrize("rel", [
    "data/processed/splits.json",
    "artifacts/x.onnx",
    "mlruns/0/meta.yaml",
    "docs/proposal.md",
    ".claude/settings.json",
    ".claude/hooks/guard_edits.py",
])
def test_protected_paths_blocked(tmp_path, rel):
    assert guard.check(_ev(tmp_path, rel), str(tmp_path)) is not None


@pytest.mark.parametrize("rel", ["src/docguard/kie/rules.py", "tests/test_x.py", "intent/001-x/spec.md",
                                 "docs/process/ai-native-sdlc.md", "database.py"])
def test_normal_paths_allowed(tmp_path, rel):
    assert guard.check(_ev(tmp_path, rel), str(tmp_path)) is None


def test_test_lock_blocks_tests_only(tmp_path):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "test-lock").touch()
    assert guard.check(_ev(tmp_path, "tests/test_x.py"), str(tmp_path)) is not None
    assert guard.check(_ev(tmp_path, "src/docguard/x.py"), str(tmp_path)) is None


@pytest.mark.parametrize("secret", [
    "-----BEGIN RSA PRIVATE KEY-----",
    "token = 'ghp_" + "a" * 36 + "'",
    "HF_TOKEN=hf_" + "b" * 34,
])
def test_secret_content_blocked(tmp_path, secret):
    assert guard.check(_ev(tmp_path, "src/x.py", content=secret), str(tmp_path)) is not None
    assert guard.check(_ev(tmp_path, "src/x.py", new_string=secret), str(tmp_path)) is not None


@pytest.mark.skipif(os.name != "nt", reason="Git-Bash path form only occurs on Windows")
def test_msys_style_root(tmp_path):
    drive, rest = os.path.splitdrive(str(tmp_path))
    msys_root = "/" + drive[0].lower() + rest.replace("\\", "/")
    assert guard.check(_ev(tmp_path, "data/a.csv"), msys_root) is not None
    assert guard.check(_ev(tmp_path, "src/a.py"), msys_root) is None


def test_cli_exit_codes(tmp_path):
    env = {**os.environ, "CLAUDE_PROJECT_DIR": str(tmp_path)}
    blocked = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(_ev(tmp_path, "data/a.csv")),
                             capture_output=True, text=True, env=env)
    assert blocked.returncode == 2 and "data/a.csv" in blocked.stderr
    ok = subprocess.run([sys.executable, str(HOOK)], input=json.dumps(_ev(tmp_path, "src/a.py")),
                        capture_output=True, text=True, env=env)
    assert ok.returncode == 0
