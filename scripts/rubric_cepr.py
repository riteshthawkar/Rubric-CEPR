#!/usr/bin/env python3
"""Run from a source checkout; editable installation is optional."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rubric_cepr.cli import main

if __name__ == "__main__":
    main()
