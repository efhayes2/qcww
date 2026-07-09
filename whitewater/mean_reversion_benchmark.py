"""Mean-reversion benchmark for the Waha basis spreads.

Motivation
----------
The XGBoost strategies do not forecast the spread out-of-sample: gradient-boosted
trees cannot extrapolate across gas-price regimes, so `pred` collapses to a near-
constant (~training mean) and the trading signal `sign(pred - spread)` degenerates
into a crude revert-to-a-fixed-level rule. A plain z-score band on the spread's own
history captures the same mean-reversion more cheaply and with far less turnover.

This module is that honest benchmark. It reuses the spot strategy's data prep so the
spreads, weather-based sizing, and evaluation window match the ML runs exactly.

Transaction costs (as specified by the desk)
--------------------------------------------
`tc_per_mmbtu` is a ROUND-TRIP cost, charged half on entry and half on exit. Holding
the same direction is free even when the position is resized (you never cross the
spread). So cost is incurred only on a sign flip or a move to/from flat:
    flip long->short : tc/2*|old| (exit) + tc/2*|new| (enter)
    long->long       : 0   (also short->short, and any same-direction resize)
    first entry      : tc/2*|pos|;   final exit: tc/2*|pos|
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

from whitewater.gbm_trading_strategy_2 import WhitewaterSpotStrategy
from whitewater.costs import transaction_cost


class WahaMeanReversionBenchmark:
    """Z-score band mean-reversion on the Waha basis spreads.

    Enter short when the spread is rich (z > entry_z), long when cheap (z < -entry_z),
    and flatten when it reverts back inside +/- exit_z. Position size reuses the spot
    strategy's cold-snap sizing so the only thing that differs from the ML run is the
    signal itself.
    """

    def __init__(self, excel_path, base_lots=3, high_vol_lots=10, tc_per_mmbtu=0.05,
                 exclude_hh=False, weather_mode='today',
                 window=60, entry_z=1.5, exit_z=0.25,
                 start='2021-01-01', end='2024-12-31', mask_uri=True):
        self.window, self.entry_z, self.exit_z = window, entry_z, exit_z
        self.start, self.end, self.mask_uri = start, end, mask_uri
        self.tc_per_mmbtu = tc_per_mmbtu
        # Borrow the spot strategy purely for its data prep (spreads + weather sizing column).
        self._spot = WhitewaterSpotStrategy(excel_path, base_lots, high_vol_lots,
                                            tc_per_mmbtu, exclude_hh, weather_mode)
        self.spreads = self._spot.spreads
        self.contract_size = self._spot.contract_size
        self.base_lots, self.high_vol_lots = base_lots, high_vol_lots
        self.df = None
        self.results_df = None

    def load_and_prep(self):
        self._spot.load_and_prep()
        self.df = self._spot.full_df.loc[self.start:self.end].copy()
        return self

    def _band_state(self, spread):
        """Position state in {-1, 0, +1} from a causal z-score band (no look-ahead)."""
        x = self.df[spread]
        z = (x - x.rolling(self.window).mean()) / x.rolling(self.window).std()
        state = np.zeros(len(x))
        cur = 0
        for i, zt in enumerate(z.values):
            if np.isnan(zt):
                cur = 0
            elif cur == 0:
                cur = -1 if zt > self.entry_z else (1 if zt < -self.entry_z else 0)
            elif cur == 1 and zt > -self.exit_z:   # long has reverted toward the mean
                cur = 0
            elif cur == -1 and zt < self.exit_z:   # short has reverted toward the mean
                cur = 0
            state[i] = cur
        return pd.Series(state, index=x.index)

    def run(self):
        df = self.df
        lots = np.where(df['weather_signal_MAF'] < 35, self.high_vol_lots, self.base_lots)
        uri = (df.index >= '2021-02-01') & (df.index <= '2021-02-28')

        for s in self.spreads:
            state = self._band_state(s)
            pos = state * lots * self.contract_size
            gross = df[s].diff().shift(-1) * pos
            cost = transaction_cost(pos, self.tc_per_mmbtu)
            pl = (gross - cost).fillna(0.0)
            if self.mask_uri:
                pl[uri] = 0.0
            df[f'state_{s}'] = state
            df[f'daily_pl_{s}'] = pl
            df[f'tc_{s}'] = cost

        df['total_daily_pl'] = df[[f'daily_pl_{s}' for s in self.spreads]].sum(axis=1)
        df['nav'] = df['total_daily_pl'].cumsum()
        self.results_df = df
        return self

    def metrics(self):
        pl = self.results_df['total_daily_pl']
        nav = self.results_df['nav']
        flips = np.mean([(np.sign(self.results_df[f'state_{s}']).diff().fillna(0) != 0).sum()
                         for s in self.spreads])
        return {
            "Net P&L": pl.sum(),
            "Sharpe (daily-ann)": np.sqrt(252) * pl.mean() / pl.std(),
            "Max Drawdown": (nav - nav.cummax()).min(),
            "TC paid": sum(self.results_df[f'tc_{s}'].sum() for s in self.spreads),
            "Flips per spread": flips,
        }

    def export(self, save_dir='~/data/pngs/'):
        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_df.to_csv(os.path.join(save_dir, f"waha_mr_benchmark_{ts}.csv"))
        self.results_df.to_csv(os.path.join(save_dir, "waha_mr_benchmark_current.csv"))
        return self


if __name__ == "__main__":
    from pathlib import Path
    XLSX = str(Path(__file__).resolve().parent / "data" / "trading_data.xlsx")

    bench = WahaMeanReversionBenchmark(XLSX, window=60, entry_z=1.5, exit_z=0.25,
                                       tc_per_mmbtu=0.05, exclude_hh=False)
    bench.load_and_prep().run().export()

    print("Waha mean-reversion benchmark  (z-band k=60, entry=1.5, exit=0.25, tc=$0.05 round-trip)")
    print("-" * 60)
    for k, v in bench.metrics().items():
        print(f"  {k:<20} {v:>14,.2f}" if 'Sharpe' in k or 'Flips' in k
              else f"  {k:<20} ${v:>14,.0f}")
