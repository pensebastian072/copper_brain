"""Copper Brain CLI. P1 ships `ingest`; P2+ add score/publish/update verbs.

    python -m copper_brain.cli ingest -v     # pull all sources, write clean+freshness
    python -m copper_brain.cli freshness     # print the last freshness manifest
"""
from __future__ import annotations

import argparse
import json
import sys

from . import config
from .ingest import run_ingest


def cmd_ingest(args) -> int:
    combined, _records = run_ingest(verbose=True)
    rows = 0 if combined is None else len(combined)
    return 0 if rows > 0 else 1


def cmd_freshness(_args) -> int:
    if not config.FRESHNESS_FILE.exists():
        print("no freshness.json yet — run `ingest` first", file=sys.stderr)
        return 1
    print(config.FRESHNESS_FILE.read_text())
    return 0


def cmd_regime(_args) -> int:
    """Print the current published regime (via the fail-safe reader)."""
    from .publish import read_regime
    print(json.dumps(read_regime(), indent=2))
    return 0


def cmd_ui(args) -> int:
    """Launch the local dashboard (127.0.0.1 only)."""
    from .ui_server import main as ui_main
    return ui_main(["--port", str(args.port)])


def cmd_gate(args) -> int:
    """Evaluate a copper alert (JSON file or inline) against the regime."""
    from .webhook_gate import evaluate_alert
    if args.alert:
        alert = json.loads(args.alert)
    elif args.file:
        alert = json.loads(open(args.file).read())
    else:
        print("provide --alert '<json>' or --file <path>", file=sys.stderr)
        return 2
    decision = evaluate_alert(alert)
    print(json.dumps(decision, indent=2))
    return 0 if decision.get("confirm") else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="copper_brain")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="pull all sources -> clean + freshness")
    p_ing.add_argument("-v", "--verbose", action="store_true")
    p_ing.set_defaults(func=cmd_ingest)

    p_fr = sub.add_parser("freshness", help="print last freshness manifest")
    p_fr.set_defaults(func=cmd_freshness)

    p_rg = sub.add_parser("regime", help="print current published regime (fail-safe)")
    p_rg.set_defaults(func=cmd_regime)

    p_ui = sub.add_parser("ui", help="launch the local dashboard (127.0.0.1)")
    p_ui.add_argument("--port", type=int, default=8077)
    p_ui.set_defaults(func=cmd_ui)

    p_gt = sub.add_parser("gate", help="evaluate a copper alert against the regime")
    p_gt.add_argument("--alert", help="inline alert JSON string")
    p_gt.add_argument("--file", help="path to an alert JSON file")
    p_gt.set_defaults(func=cmd_gate)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
