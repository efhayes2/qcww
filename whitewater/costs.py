"""Shared transaction-cost model for whitewater basis backtests.

`tc_per_mmbtu` is a ROUND-TRIP cost, charged half on entry and half on exit. Holding
the same direction is free even when the position is resized (you never cross the
spread). Cost is incurred only on a sign flip or a move to/from flat:
    flip long->short : tc/2*|old| (exit) + tc/2*|new| (enter)
    long->long       : 0   (also short->short, and any same-direction resize)
    first entry      : tc/2*|pos|;   final exit: tc/2*|pos|
"""
import numpy as np
import pandas as pd


def transaction_cost(pos, tc):
    """Round-trip tc split half-on-entry / half-on-exit; same-direction continuation is free.

    pos : signed position in MMBtu (pandas Series, one row per day, chronological).
    tc  : round-trip cost per MMBtu.
    Returns a per-day cost Series aligned to `pos`.
    """
    prev = pos.shift(1).fillna(0.0)
    same_dir = (np.sign(pos) == np.sign(prev)) & (np.sign(pos) != 0)
    exit_c = np.where(same_dir, 0.0, (tc / 2) * prev.abs())   # close old book unless we hold
    entry_c = np.where(same_dir, 0.0, (tc / 2) * pos.abs())   # open new book unless we hold
    cost = pd.Series(exit_c + entry_c, index=pos.index)
    cost.iloc[-1] += (tc / 2) * abs(pos.iloc[-1])             # pay to flatten the final position
    return cost
