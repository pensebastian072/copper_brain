"""Skip @pytest.mark.network tests unless explicitly selected.

Run live source tests with:  python -m pytest -m network
The default `python -m pytest` stays offline, fast, deterministic.
"""
import pytest


def pytest_collection_modifyitems(config, items):
    selected = config.getoption("-m")
    if "network" in (selected or ""):
        return  # user asked for network tests; run them
    skip = pytest.mark.skip(reason="network test; run with -m network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
