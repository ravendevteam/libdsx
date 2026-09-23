from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "read_dsx.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def run_cli(*arguments: str | Path, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(argument) for argument in arguments)],
        cwd=cwd,
        capture_output=True,
        check=False,
        timeout=20,
    )


def test_cli_prints_entire_specification_with_lf_endings(tmp_path, trace) -> None:
    trace("Run the project-root reader script from a separate working directory")
    result = run_cli(FIXTURES / "DossierRev1.dsx", cwd=tmp_path)
    expected = (FIXTURES / "DossierRev1.txt").read_text(encoding="ascii").encode("ascii")
    assert result.returncode == 0
    assert result.stderr == b""
    assert result.stdout == expected
    assert b"\r" not in result.stdout
    trace(f"CLI printed the complete {len(expected)}-byte canonical document")


def test_cli_accepts_filename_with_spaces(tmp_path) -> None:
    destination = tmp_path / "document with spaces.dsx"
    destination.write_bytes((FIXTURES / "DossierRev1.dsx").read_bytes())
    result = run_cli(destination.name, cwd=tmp_path)
    expected = (FIXTURES / "DossierRev1.txt").read_text(encoding="ascii").encode("ascii")
    assert result.returncode == 0
    assert result.stderr == b""
    assert result.stdout == expected


def test_cli_reports_missing_argument_on_stderr(tmp_path) -> None:
    result = run_cli(cwd=tmp_path)
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr


def test_cli_reports_missing_file_without_partial_output(tmp_path) -> None:
    result = run_cli(tmp_path / "missing.dsx", cwd=tmp_path)
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr


def test_cli_rejects_plaintext_without_partial_output(tmp_path) -> None:
    destination = tmp_path / "invalid.dsx"
    destination.write_bytes(b"A plain text document.\n")
    result = run_cli(destination, cwd=tmp_path)
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr


def test_cli_validates_entire_content_before_printing(tmp_path) -> None:
    encoded = bytearray((FIXTURES / "DossierRev1.dsx").read_bytes())
    encoded[-1] ^= 1
    destination = tmp_path / "corrupted-content.dsx"
    destination.write_bytes(encoded)
    result = run_cli(destination, cwd=tmp_path)
    assert result.returncode != 0
    assert result.stdout == b""
    assert result.stderr
