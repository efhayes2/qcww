# START_HERE_Q — Whitewater basis strategies (resume notes)

_Last worked: 2026-07-09. All code committed + pushed to `origin/main` (last commit `2d81e35`)._

## Bottom line (read this first)
After stripping out four sources of fake performance, **the only genuine, tradeable, cost-robust
edge is a roll-aware z-band mean-reversion on the M1→M2 swap spreads: ~0.57–0.67 Sharpe.**
The XGBoost strategies have **no real edge**:
- **Spot** is non-tradeable — physical delivery doesn't settle against the next day's spot, so its
  `diff`-based P&L is fictional. **We are not trading the spot idea.**
- **Forward XGB** goes **negative** (−0.72 at 1¢ … −1.26 at 5¢) once P&L is booked on the contract
  actually held. Its apparent +0.75 was entirely the monthly roll (expiry) jump.

## What got fixed this session (each removed fake P&L)
1. **Paths** — `QuantCode26` → `__file__`-relative (repo was renamed to `qcww`).
2. **Weather look-ahead** — `weather_mode` flag: `'today'` (forecast proxy, no look-ahead, default)
   vs `'tomorrow'` (perfect-foresight benchmark). Was worth ~0.33 Sharpe on forward, not the whole edge.
3. **Transaction costs** — old code charged `|position|*tc` *every day* (turnover-blind carry).
   Now `whitewater/costs.py::transaction_cost`: round-trip tc, **half on entry / half on exit,
   same-direction continuation (incl. resize) free**. This is the desk convention.
4. **Roll artifact (biggest)** — the M1/M2 swap columns relabel at NG expiry (~1 BD before EOM,
   inside bid week). Trading continuous M1 books that expiry jump as fake profit. Fixed via
   `build_rolled_forward_spreads` (in `gbm_strategy_on_forwards.py`): hold M1, roll the position
   into M2 at **EOM−5BD** (bid week), follow that same contract through the relabel → every daily
   return is within one contract. Both the forward XGB (`simulate_forward_logic`, and the unified
   copy) and the MR benchmark now book P&L this way. Signal is still M1-based `sign(pred−spread)`;
   only P&L accounting changed. Confirmed: roll is a fixed NG calendar shared across all hubs.

## The real benchmark
`whitewater/mean_reversion_benchmark.py` → `WahaMeanReversionBenchmark(source='forward')`
- z-score band: window=60, entry_z=1.5, exit_z=0.25; enter when |z|≥1.5, flatten near the mean.
- `roll_aware=True` (default for forward), `roll_bd=5`, `handoff_bd=1`.
- Result (2022–2024, `today` mode, tc=$0.05): **Sharpe 0.57, Net $1.87M, MaxDD −$1.0M, 31 flips/spread.**
- 60-day σ intentionally spans ~3 contracts (same-region basis — desk endorsed).

## How to run (no pip/pandas on this box)
Bootstrap a venv (system python is PEP-668 locked, no sudo):
```
python3 -m venv --without-pip <dir> && <dir>/bin/python get-pip.py
<dir>/bin/pip install pandas numpy xgboost scikit-learn matplotlib openpyxl
```
Then run as package modules from the repo root, headless:
```
MPLBACKEND=Agg <venv>/bin/python -m whitewater.mean_reversion_benchmark
MPLBACKEND=Agg <venv>/bin/python -m whitewater.gbm_strategy_on_forwards
MPLBACKEND=Agg <venv>/bin/python -m whitewater.gbm_trading_strategy_2   # spot (non-tradeable, kept for reference)
```
Outputs (CSVs, dashboard PNGs, tearsheet PDFs) land in `~/data/pngs/` (outside the repo).

## Next ideas (priority order)
1. **Spot-basis change as a lead-lag signal into the forward.** Physical basis often leads the swap
   curve. This is the most promising unexplored edge to *stack on* the MR baseline. Not done anywhere
   today — forward XGB only uses physical price *levels*, never the spot spread or its change.
2. **z-band robustness sweep** (window / entry_z / exit_z) — how sensitive is the ~0.6 Sharpe?
3. **Exact roll calendar** — `handoff_bd=1` is modal; off by ±1 day in ~1/3 of months (tiny P&L
   impact). Wire in the per-month detected roll date only if exactness is needed.
4. `whitewater_forward_current.csv` is a single "current" pointer that does NOT encode `weather_mode`
   (profit_attribution reads that hardcoded path) — split if you compare modes.

## Key files
- `whitewater/mean_reversion_benchmark.py` — the benchmark + `build_rolled_forward_spreads` usage.
- `whitewater/gbm_strategy_on_forwards.py` — forward XGB + `build_rolled_forward_spreads` (roll logic).
- `whitewater/costs.py` — `transaction_cost` (desk convention).
- `whitewater/gbm_trading_strategy_2.py` — spot XGB (non-tradeable; reference only).
- `whitewater/whitewater.db` — 11 tables: `phys_prices/weather/lng`, `swap_*` (M1–M6). M2/M3 fully populated.
