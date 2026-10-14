"""Comments must not cite identifiers or paths that have ceased to exist.

This is the defect class that hid three real bugs earlier in the project: a comment describing code that had moved on.
One of them had swallowed a live statement, one guarded a precedence error, and one indexed a value its own guard no
longer defined. None was visible by reading the comment alone, and all three were survivable for months.

So the check is a test rather than a one-off. Two mechanical rules, chosen because they cannot produce false positives
that a reader would have to argue with:

  1. a backticked identifier in a comment whose name appears nowhere in the package, scripts, tests, configs or docs;
  2. a file path in a comment that does not exist.

Prose and numbers are deliberately not checked. Neither can be verified mechanically, and a false failure here would
train a reader to ignore the test.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKTICK = re.compile(r"`([^`]+)`")
PATHLIKE = re.compile(
    r"\b((?:scripts|parakeet|tests|configs|docs|runs|data)/[A-Za-z0-9_./-]+\.(?:py|yaml|json|md|jsonl|pt))"
)
SHORT = {"x", "y", "t", "n", "c", "k", "b", "e", "i", "j"}


def _sources() -> list[Path]:
    return sorted(
        list((ROOT / "parakeet").rglob("*.py"))
        + list((ROOT / "scripts").glob("*.py"))
        + list((ROOT / "tests").glob("*.py"))
    )


def _universe() -> str:
    """Everything the codebase mentions, so a miss means the name exists nowhere at all."""
    parts = [path.read_text(encoding="utf-8") for path in _sources()]
    for folder, suffixes in (("docs", {".md"}), ("configs", {".yaml"})):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def _probe(token: str) -> str | None:
    token = token.strip()
    if not (2 <= len(token) <= 40):
        return None
    if token.startswith("--"):
        return token[2:].replace("-", "_")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", token):
        return token.split(".")[-1]
    return None


def test_no_comment_cites_an_identifier_that_exists_nowhere():
    universe = _universe()
    missing = []
    for path in _sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip().startswith("#"):
                continue
            for token in BACKTICK.findall(line):
                probe = _probe(token)
                if probe and probe not in SHORT and probe not in universe:
                    missing.append(f"{path.relative_to(ROOT).as_posix()}:{number} `{token}`")
    assert not missing, "comments cite names that exist nowhere: " + "; ".join(missing)


def test_no_comment_cites_a_path_that_does_not_exist():
    missing = []
    for path in _sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip().startswith("#"):
                continue
            for match in PATHLIKE.findall(line):
                if not (ROOT / match).exists():
                    missing.append(f"{path.relative_to(ROOT).as_posix()}:{number} {match}")
    assert not missing, "comments cite missing paths: " + "; ".join(missing)
