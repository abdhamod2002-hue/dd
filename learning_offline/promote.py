"""Atomic, versioned promotion of candidate YOLO weights after gate pass."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from learning_offline import CANDIDATE_DIR, PRODUCTION_WEIGHTS


def promote_weights(
    candidate_pt: Path,
    *,
    decision: Optional[Dict[str, Any]] = None,
    production: Path = PRODUCTION_WEIGHTS,
) -> Dict[str, Any]:
    """Archive current ``best.pt`` as ``best.pt.vN`` then replace with candidate.

    Never runs unless the caller already passed ``should_promote``.
    """
    candidate_pt = Path(candidate_pt)
    if not candidate_pt.is_file():
        raise FileNotFoundError(candidate_pt)
    production = Path(production)
    production.parent.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = None
    if production.is_file():
        # Next version index
        existing = sorted(production.parent.glob("best.pt.v*"))
        next_idx = len(existing) + 1
        archived = production.parent / f"best.pt.v{next_idx}"
        shutil.copy2(production, archived)

    staging = production.with_suffix(".pt.promoting")
    shutil.copy2(candidate_pt, staging)
    staging.replace(production)

    meta = {
        "promoted_at": stamp,
        "candidate": str(candidate_pt),
        "production": str(production),
        "archived_previous": str(archived) if archived else None,
        "gate": decision or {},
    }
    meta_path = production.parent / "promotion_log.jsonl"
    with meta_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(meta) + "\n")
    return meta
