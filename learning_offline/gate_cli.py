#!/usr/bin/env python3
"""CLI: compare two frozen_eval reports and optionally promote candidate weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from learning_offline.gate import should_promote_from_files  # noqa: E402
from learning_offline.promote import promote_weights  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Motared self-learning promotion gate")
    ap.add_argument("--current-report", required=True, help="frozen_eval JSON for production weights")
    ap.add_argument("--candidate-report", required=True, help="frozen_eval JSON for candidate weights")
    ap.add_argument("--candidate-weights", default="", help="path to candidate .pt (required with --promote)")
    ap.add_argument("--fpr-slack", type=float, default=1.1)
    ap.add_argument("--promote", action="store_true", help="If gate passes, swap into best.pt (versioned)")
    args = ap.parse_args()

    decision = should_promote_from_files(
        args.candidate_report,
        args.current_report,
        fpr_slack=float(args.fpr_slack),
    )
    print(json.dumps(decision.to_dict(), indent=2))

    if not decision.promote:
        print("[gate] NOT promoting")
        return 1

    if args.promote:
        if not args.candidate_weights:
            print("[gate] --candidate-weights required with --promote", file=sys.stderr)
            return 2
        meta = promote_weights(Path(args.candidate_weights), decision=decision.to_dict())
        print("[gate] PROMOTED:", json.dumps(meta, indent=2))
    else:
        print("[gate] would promote (pass --promote to swap weights)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
