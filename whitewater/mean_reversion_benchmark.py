"""Mean-reversion benchmark for the Waha basis spreads.

Motivation
----------
The XGBoost strategies do not forecast the spread out-of-sample: gradient-boosted
trees cannot extrapolate across gas-price regimes, so `pred` collapses to a near-
constant (~training mean) and the trading signal `sign(pred - spread)` degenerates
into a crude revert-to-a-fixed-level rule. A plain z-score band on the spread's own
history captures the same mean-reversion more cheaply and with far less turnover.

Instrument (`source`)
---------------------
- 'forward' (default): trade the M1 swap spreads (HH_M1 - WA_M1, etc.). These are
  financial and settle/mark daily, so `spread.diff().shift(-1)` P&L is real and
  tradeable. This is the universe we actually care about.
- 'spot': trade the physical basis (Henry_Hub - Waha, etc.). Physical spot is
  delivery, not a next-day-settling instrument, so its diff-based P&L is NOT
  tradeable -- kept only as a relative signal-quality comparison vs the spot XGB.

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
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

from whitewater.gbm_trading_strategy_2 import WhitewaterSpotStrategy
from whitewater.gbm_strategy_on_forwards import WhitewaterForwardStrategy
from whitewater.costs import transaction_cost


def build_rolled_forward_spreads(db_path, roll_bd=5, handoff_bd=1):
    """Within-contract Waha basis spreads from the M1/M2 swap curves.

    The raw M1/M2 columns are a rolling term structure whose underlying contract relabels
    at NG expiry (~`handoff_bd` BD before EOM), *inside* bid week. Trading the continuous M1
    series therefore books that expiry jump as fake P&L. Instead we hold the front (M1) most
    of the month, roll the position into the next contract (M2) at EOM-`roll_bd` BD (bid week
    opens), and follow that same contract through the relabel (M2 -> M1) so every daily return
    is within a single contract. Same NG-linked roll calendar for all hubs.

    Returns a DataFrame indexed by date with, per spread s in {hh, katy, hsc}:
      target_waha_s     : held spread level (drives the z-score signal)
      ret_target_waha_s : within-contract daily spread change (drives P&L)
    """
    tbl = {'HH': 'swap_hh', 'WA': 'swap_wa', 'KT': 'swap_kt', 'HS': 'swap_hs'}
    conn = sqlite3.connect(str(db_path))
    legs = {k: pd.read_sql(f"SELECT date,M1,M2 FROM {t}", conn, parse_dates=['date'])
                 .set_index('date').sort_index() for k, t in tbl.items()}
    conn.close()

    idx = legs['HH'].index
    legs = {k: v.reindex(idx).ffill() for k, v in legs.items()}
    bfe = pd.Series(range(len(idx)), index=idx).groupby(idx.to_period('M')).transform(
        lambda s: s.max() - s)                                  # 0 = last trading day of month
    col = pd.Series(np.where((bfe > handoff_bd) & (bfe <= roll_bd), 'M2', 'M1'), index=idx)

    def held(leg):
        return pd.Series(np.where(col == 'M2', leg['M2'], leg['M1']), index=idx)

    def within_ret(leg):
        # Return over [t-1, t] on the contract held entering the period (col at t-1). At the
        # relabel (prev M2 -> now M1) it is the same contract, so read M1_t vs M2_{t-1}. On the
        # position-roll day (prev M1 -> now M2) we still book the OLD front's move (M1_t vs M1_{t-1});
        # switching to M2 is a trade, not a marked jump.
        pc = col.shift(1)
        prev = np.where(pc == 'M2', leg['M2'].shift(1), leg['M1'].shift(1))
        now = np.where(pc == 'M2', np.where(col == 'M1', leg['M1'], leg['M2']), leg['M1'])
        return pd.Series(now - prev, index=idx)

    hp = {k: held(v) for k, v in legs.items()}
    wr = {k: within_ret(v) for k, v in legs.items()}
    out = pd.DataFrame(index=idx)
    for s, h in {'hh': 'HH', 'katy': 'KT', 'hsc': 'HS'}.items():
        out[f'target_waha_{s}'] = hp[h] - hp['WA']
        out[f'ret_target_waha_{s}'] = wr[h] - wr['WA']
    return out


class WahaMeanReversionBenchmark:
    """Z-score band mean-reversion on the Waha basis spreads.

    Enter short when the spread is rich (z > entry_z), long when cheap (z < -entry_z),
    and flatten when it reverts back inside +/- exit_z. Position sizing reuses the
    underlying strategy's cold-snap sizing so the only thing that differs from the ML
    run is the signal itself.
    """

    def __init__(self, data_path, source='forward',
                 base_lots=None, high_vol_lots=None, tc_per_mmbtu=0.05,
                 exclude_hh=False, weather_mode='today',
                 window=60, entry_z=1.5, exit_z=0.25,
                 start=None, end='2024-12-31', mask_uri=None,
                 roll_aware=None, roll_bd=5, handoff_bd=1):
        if source not in ('forward', 'spot'):
            raise ValueError("source must be 'forward' or 'spot'")
        self.source = source
        self.window, self.entry_z, self.exit_z = window, entry_z, exit_z
        self.tc_per_mmbtu = tc_per_mmbtu
        # Roll-aware within-contract P&L only applies to the forward (swap) source.
        self.roll_aware = (source == 'forward') if roll_aware is None else roll_aware
        self.roll_bd, self.handoff_bd = roll_bd, handoff_bd

        # Sizing defaults mirror each strategy's own scale. Base:high ratio is 1:3.33 in both,
        # so Sharpe is identical either way; the absolute lots just match the corresponding
        # XGB run's dollar P&L.
        if source == 'forward':
            base_lots = 15 if base_lots is None else base_lots
            high_vol_lots = 50 if high_vol_lots is None else high_vol_lots
            self._strat = WhitewaterForwardStrategy(data_path, base_lots, high_vol_lots,
                                                    tc_per_mmbtu, weather_mode)
            start = '2022-01-01' if start is None else start   # M1 swap coverage; matches XGB forward
            mask_uri = False if mask_uri is None else mask_uri  # Uri (Feb-21) precedes this window
        else:
            base_lots = 3 if base_lots is None else base_lots
            high_vol_lots = 10 if high_vol_lots is None else high_vol_lots
            self._strat = WhitewaterSpotStrategy(data_path, base_lots, high_vol_lots,
                                                 tc_per_mmbtu, exclude_hh, weather_mode)
            start = '2021-01-01' if start is None else start
            mask_uri = True if mask_uri is None else mask_uri

        self.start, self.end, self.mask_uri = start, end, mask_uri
        self.base_lots, self.high_vol_lots = base_lots, high_vol_lots
        self.spreads = self._strat.spreads
        self.contract_size = self._strat.contract_size
        self.df = None
        self.results_df = None

    def load_and_prep(self):
        self._strat.load_and_prep()
        if self.roll_aware:
            # Roll-aware forward: within-contract spreads from the M1/M2 curves, plus the
            # weather sizing column carried over from the strategy's prep.
            rolled = build_rolled_forward_spreads(self._strat.db_path, self.roll_bd, self.handoff_bd)
            wx = self._strat.full_df[['weather_signal_MAF']]
            self.df = rolled.join(wx, how='inner').loc[self.start:self.end].copy()
        else:
            self.df = self._strat.full_df.loc[self.start:self.end].copy()
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
            # Roll-aware P&L uses the within-contract return; otherwise the level diff.
            spread_ret = df[f'ret_{s}'] if f'ret_{s}' in df.columns else df[s].diff()
            gross = spread_ret.shift(-1) * pos
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
        self.results_df.to_csv(os.path.join(save_dir, f"waha_mr_{self.source}_{ts}.csv"))
        self.results_df.to_csv(os.path.join(save_dir, f"waha_mr_{self.source}_current.csv"))
        return self


if __name__ == "__main__":
    from pathlib import Path
    DB = str(Path(__file__).resolve().parent / "whitewater.db")

    # Tradeable benchmark: mean-reversion on the M1 swap spreads.
    bench = WahaMeanReversionBenchmark(DB, source='forward', window=60, entry_z=1.5, exit_z=0.25,
                                       tc_per_mmbtu=0.05)
    bench.load_and_prep().run().export()

    print(f"Waha MR benchmark [{bench.source} / M1->M2 swaps, roll_aware={bench.roll_aware}]  "
          f"(z-band k=60, entry=1.5, exit=0.25, tc=$0.05 round-trip, {bench.start}..{bench.end})")
    print("-" * 60)
    for k, v in bench.metrics().items():
        print(f"  {k:<20} {v:>14,.2f}" if 'Sharpe' in k or 'Flips' in k
              else f"  {k:<20} ${v:>14,.0f}")
