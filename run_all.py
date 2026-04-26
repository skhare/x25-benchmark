"""
Run the full benchmark pipeline end-to-end.

Phase 1 (data collection) needs OPENROUTER_API_KEY.
Phase 2 (scoring)         needs OPENAI_API_KEY for the xsum and rag judge.
Phases 3 + 4 are pure replay/charts and need no keys.

Usage:
    python benchmark/run_all.py                   # all phases
    python benchmark/run_all.py --skip 1          # phases 2-4 only
    python benchmark/run_all.py --only 3 4        # phases 3 and 4 only
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, nargs="*", default=[])
    ap.add_argument("--only", type=int, nargs="*", default=[])
    args = ap.parse_args()

    phases = {
        1: ("collect raw data",   "phase1_collect",   "collect"),
        2: ("score responses",    "phase2_score",     "score_all"),
        3: ("simulate strategies","phase3_strategies","simulate"),
        4: ("charts & summary",   "phase4_charts",    "make_all"),
    }
    selected = sorted(args.only) if args.only else [n for n in phases if n not in args.skip]

    for n in selected:
        title, mod_name, fn_name = phases[n]
        print()
        print("=" * 70)
        print(f"PHASE {n}: {title}")
        print("=" * 70)
        mod = __import__(mod_name)
        getattr(mod, fn_name)()


if __name__ == "__main__":
    main()
