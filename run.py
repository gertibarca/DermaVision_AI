"""Nis DermaVision AI dhe hap automatikisht shfletuesin."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli

PROJECT_DIR = Path(__file__).resolve().parent


def main() -> int:
    os.chdir(PROJECT_DIR)
    sys.argv = [
        "streamlit",
        "run",
        str(PROJECT_DIR / "app.py"),
        "--server.headless=false",
        "--server.port=8501",
    ]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
