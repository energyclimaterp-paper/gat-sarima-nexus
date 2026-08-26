"""Benchmark forecast backends on THIS machine, CPU-only.

Synthetic monthly series shaped like the notebook's targets (trend + 12-month
seasonality + noise, 96 months, 12-month holdout). Reports accuracy relative to
seasonal-naive (MASE-style) plus wall-clock, which is what decides PC vs Kaggle.
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from nexus_local import naive_forecast, chronos_forecast, load_chronos

RNG = np.random.default_rng(42)
H, N_SERIES, MONTHS = 12, 24, 96

def make_series(i):
    t = np.arange(MONTHS)
    level = RNG.uniform(500, 40000)
    trend = RNG.uniform(-0.004, 0.010) * level * t
    seas = 0.18 * level * np.sin(2 * np.pi * (t % 12) / 12 + RNG.uniform(0, 6.28))
    noise = RNG.normal(0, 0.04 * level, MONTHS)
    v = np.clip(level + trend + seas + noise, 1, None)
    idx = pd.date_range("2017-01-01", periods=MONTHS, freq="MS")
    return pd.Series(v, index=idx)

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))

def sarima_forecast(train, n):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    r = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,0,12),
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
    return r.get_forecast(n).predicted_mean.values

series = [make_series(i) for i in range(N_SERIES)]
splits = [(s.iloc[:-H], s.iloc[-H:]) for s in series]

print(f"{N_SERIES} synthetic monthly series x {MONTHS} months, {H}-month holdout, CPU only\n")

print("warming chronos (first call includes model load)...", flush=True)
t0 = time.perf_counter(); load_chronos(); load_t = time.perf_counter() - t0
print(f"  model load: {load_t:.1f}s\n")

backends = {
    "seasonal-naive": naive_forecast,
    "SARIMA(1,1,1)(1,1,0,12)": sarima_forecast,
    "chronos-2 (120M)": chronos_forecast,
}

results = {}
for name, fn in backends.items():
    errs, t0 = [], time.perf_counter()
    fails = 0
    for train, test in splits:
        try:
            errs.append(rmse(fn(train, H), test.values))
        except Exception as e:
            fails += 1; errs.append(np.nan)
    dt = time.perf_counter() - t0
    results[name] = (np.nanmean(errs), dt, fails)

base = results["seasonal-naive"][0]
print(f"{'backend':28s} {'mean RMSE':>11s} {'vs naive':>9s} {'total s':>8s} {'s/series':>9s} {'fail':>5s}")
print("-" * 76)
for name, (e, dt, f) in results.items():
    print(f"{name:28s} {e:11.1f} {e/base:8.2f}x {dt:8.2f} {dt/N_SERIES:9.3f} {f:5d}")

print(f"\nProjected wall-clock for the real run (~200 region-series, x2 for both passes):")
for name, (e, dt, f) in results.items():
    per = dt / N_SERIES
    print(f"  {name:28s} {per*200:7.1f}s  ({per*200/60:.1f} min)")
