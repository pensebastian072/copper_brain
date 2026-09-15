"""Forward-return direction labels for grading the score and (P3) training v2.

For each horizon h in config.HORIZONS_DAYS, label[t] = sign of the copper return
from t to t+h. These are FORWARD-looking by construction, so they are only ever
used to grade or train — never as a feature. Computed on the same primary copper
price the feature frame uses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def forward_returns(copper_px: pd.Series) -> pd.DataFrame:
    """DataFrame of forward % returns, one column per horizon: fwd_ret_<h>d."""
    out = pd.DataFrame(index=copper_px.index)
    for h in config.HORIZONS_DAYS:
        out[f"fwd_ret_{h}d"] = copper_px.shift(-h) / copper_px - 1.0
    return out


def forward_direction(copper_px: pd.Series) -> pd.DataFrame:
    """DataFrame of forward direction labels in {-1,0,+1}: dir_<h>d."""
    fr = forward_returns(copper_px)
    out = pd.DataFrame(index=copper_px.index)
    for h in config.HORIZONS_DAYS:
        out[f"dir_{h}d"] = np.sign(fr[f"fwd_ret_{h}d"])
    return out
