"""Local Copper Brain dashboard. Binds 127.0.0.1 ONLY (never tunnel/expose).

Separate read-only process: it renders the published artifacts, it never pulls
data or routes anything. State is assembled by ui_state.build_state and cached
briefly so a polling browser doesn't recompute the feature frame every second.

    python -m copper_brain.ui_server [--port 8077]
    open http://127.0.0.1:8077
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    from flask import Flask, jsonify, render_template
except ImportError:
    print("missing flask. install: .venv\\Scripts\\pip install flask", file=sys.stderr)
    sys.exit(1)

from .ui_state import build_state

app = Flask(__name__, template_folder="templates", static_folder="static")

_CACHE: dict = {"at": 0.0, "state": None}
_CACHE_TTL = 25.0  # seconds


def _state():
    now = time.time()
    if _CACHE["state"] is None or now - _CACHE["at"] > _CACHE_TTL:
        _CACHE["state"] = build_state()
        _CACHE["at"] = now
    return _CACHE["state"]


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/state")
def api_state():
    return jsonify(_state())


@app.route("/api/refresh")
def api_refresh():
    _CACHE["state"] = None
    return jsonify(_state())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="copper_brain.ui_server")
    ap.add_argument("--port", type=int, default=8077)
    args = ap.parse_args(argv)
    print(f"Copper Brain dashboard -> http://127.0.0.1:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
