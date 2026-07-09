import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

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
        # position-roll day (prev M1 -> now M2) we still book the OLD front's move; switching to
        # M2 is a trade, not a marked jump.
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


class WhitewaterForwardStrategy:
    def __init__(self, db_path, base_lots=15, high_vol_lots=50, tc_per_mmbtu=0.01, weather_mode='today'):
        self.db_path = Path(os.path.expanduser(db_path))
        self.base_lots = base_lots
        self.high_vol_lots = high_vol_lots
        self.contract_size = 10000
        self.tc_per_mmbtu = tc_per_mmbtu
        # 'today' = today's realized min temp as a persistence proxy for tomorrow's forecast (no
        # look-ahead, default); 'tomorrow' = realized next-day temp (perfect-foresight benchmark only).
        if weather_mode not in ('today', 'tomorrow'):
            raise ValueError("weather_mode must be 'today' or 'tomorrow'")
        self.weather_mode = weather_mode
        self.spreads = ['target_waha_hh', 'target_waha_katy', 'target_waha_hsc']
        self.full_df = None
        self.results_df = None

    def load_and_prep(self):
        """Loads everything directly from SQLite."""
        conn = sqlite3.connect(self.db_path)

        # 1. Physical Features
        price_df = pd.read_sql("SELECT * FROM phys_prices", conn).rename(columns={'date': 'day'})
        weather_raw = pd.read_sql("SELECT * FROM phys_weather", conn).rename(columns={'date': 'day'})
        lng_df = pd.read_sql("SELECT * FROM phys_lng", conn).rename(columns={'date': 'day'})

        # 2. Financial Targets (M1 Swaps)
        swap_hubs = {'WA': 'swap_wa', 'HH': 'swap_hh', 'KT': 'swap_kt', 'HS': 'swap_hs'}
        swap_list = []
        for hub, table in swap_hubs.items():
            s_df = pd.read_sql(f"SELECT date as day, M1 as {hub}_M1 FROM {table}", conn)
            s_df['day'] = pd.to_datetime(s_df['day'])
            s_df.set_index('day', inplace=True)
            swap_list.append(s_df)
        conn.close()

        # 3. Standardize Physical Data
        for df_item in [price_df, weather_raw, lng_df]:
            df_item['day'] = pd.to_datetime(df_item['day'])
            df_item.set_index('day', inplace=True)
            df_item.sort_index(inplace=True)
            df_item.columns = [c.replace(' ', '_') for c in df_item.columns]

        # 4. Process Weather & Join
        weather_pivoted = weather_raw.pivot_table(index='day', columns='station', values='min_temp_f').add_prefix(
            'min_temp_f_')
        df = price_df.join([weather_pivoted, lng_df] + swap_list, how='inner').ffill()

        # 5. Financial Feature Engineering
        df['target_waha_hh'] = df['HH_M1'] - df['WA_M1']
        df['target_waha_katy'] = df['KT_M1'] - df['WA_M1']
        df['target_waha_hsc'] = df['HS_M1'] - df['WA_M1']

        for s in self.spreads:
            for i in range(1, 4):
                df[f'{s}_lag{i}'] = df[s].shift(i)

        # today = min temp known at decision time (day t); tomorrow = realized next-day min temp (t+1)
        df['today_min_temp_MAF'] = df['min_temp_f_MAF']
        df['tomorrow_min_temp_MAF'] = df['min_temp_f_MAF'].shift(-1)
        # Single weather driver selected by weather_mode; the only temp column fed downstream.
        df['weather_signal_MAF'] = (df['tomorrow_min_temp_MAF'] if self.weather_mode == 'tomorrow'
                                    else df['today_min_temp_MAF'])

        # Roll-aware within-contract returns for P&L (avoids booking the M1 expiry jump).
        rolled = build_rolled_forward_spreads(self.db_path)
        ret_cols = [c for c in rolled.columns if c.startswith('ret_target_waha_')]
        df = df.join(rolled[ret_cols], how='left')

        self.full_df = df.drop(columns=['Agua_Dulce', 'min_temp_f_PEQ'], errors='ignore').dropna().sort_index()
        print(
            f"DB Load Complete. Rows: {len(self.full_df)} | Range: {self.full_df.index.min().date()} to {self.full_df.index.max().date()}")
        return self

    def run_backtest(self):
        final_results = []
        # Drop targets, raw M1 swap levels, and the today/tomorrow helpers; weather_signal_MAF is the
        # single weather feature (dated per weather_mode).
        helper_cols = ('today_min_temp_MAF', 'tomorrow_min_temp_MAF')
        feature_cols = [c for c in self.full_df.columns
                        if 'target' not in c and '_M1' not in c and c not in helper_cols]

        # Running 2022-2024 to ensure swap coverage
        for trade_year in [2022, 2023, 2024]:
            train = self.full_df.loc[:f"{trade_year - 1}-12-31"]
            test = self.full_df.loc[f"{trade_year}-01-01":f"{trade_year}-12-31"]
            if train.empty or test.empty: continue

            year_outcomes = test.copy()
            for s in self.spreads:
                model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05)
                model.fit(train[feature_cols], train[s])
                year_outcomes[f'pred_{s}'] = model.predict(test[feature_cols])
            final_results.append(year_outcomes)

        self.results_df = pd.concat(final_results)
        return self

    def simulate_forward_logic(self):
        df = self.results_df
        df['current_lots'] = np.where(df['weather_signal_MAF'] < 35, self.high_vol_lots, self.base_lots)

        for s in self.spreads:
            df[f'sig_{s}'] = np.where(df[f'pred_{s}'] > df[s], 1, -1)
            df[f'pos_{s}'] = df[f'sig_{s}'] * df['current_lots'] * self.contract_size
            # P&L on the roll-aware within-contract return (not the continuous-M1 diff, which would
            # book the monthly expiry jump). Round-trip tc, half entry/half exit, holds are free.
            df[f'tc_daily_{s}'] = transaction_cost(df[f'pos_{s}'], self.tc_per_mmbtu)
            df[f'daily_pl_{s}'] = df[f'ret_{s}'].shift(-1) * df[f'pos_{s}'] - df[f'tc_daily_{s}']

        df['total_daily_pl'] = df[[f'daily_pl_{s}' for s in self.spreads]].sum(axis=1)
        df['nav'] = df['total_daily_pl'].fillna(0).cumsum()
        return self

    def generate_rolling_sharpe(self, window=60):
        capital = 1_000_000
        daily_returns = self.results_df['total_daily_pl'] / capital
        rolling_sharpe = (daily_returns.rolling(window).mean() / daily_returns.rolling(window).std()) * np.sqrt(252)

        plt.figure(figsize=(12, 5))
        rolling_sharpe.plot(color='#3498db', lw=2, label='60-Day Forward Sharpe')
        plt.axhline(y=rolling_sharpe.mean(), color='red', linestyle='--')
        plt.title('Forward Strategy Stability: Rolling 60-Day Sharpe')

        save_path = os.path.expanduser('~/data/pngs/forward_rolling_sharpe.png')
        plt.savefig(save_path)
        print(f"Forward Sharpe plot saved to: {save_path}")
        return self

    def export(self):
        # Match the directory expected by the dashboard
        save_dir = os.path.expanduser('~/data/pngs/')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save timestamped version for history
        self.results_df.to_csv(os.path.join(save_dir, f"whitewater_forward_hist_{ts}.csv"))

        # Save the STATIC reference file the dashboard script looks for
        self.results_df.to_csv(os.path.join(save_dir, "whitewater_forward_current.csv"))

        print(f"Export complete. Dashboard source 'whitewater_forward_current.csv' updated.")
        return self

if __name__ == "__main__":
    DB_PATH = str(Path(__file__).resolve().parent / "whitewater.db")
    weather_mode = 'today'  # 'today' = forecast proxy (no look-ahead); 'tomorrow' = perfect-foresight benchmark
    strategy = WhitewaterForwardStrategy(DB_PATH, weather_mode=weather_mode)
    strategy.load_and_prep() \
        .run_backtest() \
        .simulate_forward_logic() \
        .generate_rolling_sharpe() \
        .export()