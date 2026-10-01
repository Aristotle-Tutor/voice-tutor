import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_zones_import_only_what_they_are_allowed_to() -> None:
    result = subprocess.run(
        [Path(sys.executable).with_name("lint-imports")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
