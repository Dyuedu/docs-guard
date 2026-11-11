"""PreToolUse guard for Edit/Write/NotebookEdit (AI-native SDLC build-time hook).

Blocks (exit code 2, reason on stderr):
  - edits to data / results / owner-controlled targets / the guardrails themselves
  - edits under tests/ while `.claude/test-lock` exists (bug-fix flow: failing test is committed first)
  - content that looks like a credential

See docs/process/ai-native-sdlc.md §5.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import PurePosixPath

PROTECTED = [
    ("data/", "dữ liệu chỉ được tạo bằng script (scripts/), không sửa tay"),
    ("artifacts/", "artifact phải do pipeline sinh ra"),
    ("mlruns/", "kết quả MLflow không được sửa tay"),
    ("results/", "kết quả phải do pipeline sinh ra"),
    ("docs/proposal.md", "mục tiêu/ngưỡng do owner quyết định"),
    (".claude/settings.json", "agent không được tự nới guardrail"),
    (".claude/hooks/", "agent không được tự nới guardrail"),
]

SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{32,}\b"),
]


_MSYS_DRIVE = re.compile(r"^/([a-zA-Z])(/|$)")


def _native(path: str) -> str:
    """Map Git-Bash style '/c/Users/..' to 'c:/Users/..' on Windows."""
    if os.name == "nt":
        path = _MSYS_DRIVE.sub(lambda m: f"{m.group(1)}:/", path.replace("\\", "/"))
    return path


def _rel(path: str, root: str) -> str:
    p = os.path.normcase(os.path.abspath(_native(path)))
    r = os.path.normcase(os.path.abspath(_native(root)))
    try:
        rel = os.path.relpath(p, r)
    except ValueError:  # different drive on Windows
        return PurePosixPath(path.replace("\\", "/")).as_posix()
    return rel.replace("\\", "/")


def check(event: dict, root: str) -> str | None:
    """Return a block reason, or None if the edit is allowed."""
    tool_input = event.get("tool_input") or {}
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if path:
        rel = _rel(path, root)
        if not rel.startswith("../"):
            for prefix, why in PROTECTED:
                if rel == prefix.rstrip("/") or rel.startswith(prefix):
                    return f"Blocked: '{rel}' được bảo vệ — {why}."
            if rel.startswith("tests/") and os.path.exists(os.path.join(root, ".claude", "test-lock")):
                return ("Blocked: test-lock đang bật (.claude/test-lock). "
                        "Đang fix bug — chỉ sửa code, không sửa test. Owner xóa file lock khi xong.")

    text = "\n".join(
        str(tool_input.get(k, "")) for k in ("content", "new_string", "new_source")
    )
    for edit in tool_input.get("edits") or []:
        text += "\n" + str(edit.get("new_string", ""))
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            return "Blocked: nội dung có vẻ chứa secret/credential. Dùng biến môi trường, không ghi vào file."
    return None


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    root = os.environ.get("CLAUDE_PROJECT_DIR") or event.get("cwd") or os.getcwd()
    reason = check(event, root)
    if reason:
        try:
            sys.stderr.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
