#!/usr/bin/env python3
"""Start IMG_5306 re-analysis in-process (background-friendly)."""
from __future__ import annotations

import sys
from pathlib import Path

VIDEO = Path("/app/backend/backend/uploaded_videos/20260912_181751_IMG_5306.MOV")

def main() -> int:
    sys.path.insert(0, "/app")
    from project_audit.run_db_analysis_job import main as run_main
    return int(run_main(VIDEO) or 0)

if __name__ == "__main__":
    raise SystemExit(main())
