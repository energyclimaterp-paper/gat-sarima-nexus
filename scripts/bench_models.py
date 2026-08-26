"""Head-to-head of local HF time-series foundation models, CPU-only, on this box."""
import sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
from nexus_local import naive_forecast

RNG = np.random.default_rng(42)
H, N, MONTHS = 12, 24, 96

def make_series():
    t = np.arange(MONTHS); level = RNG.uniform(500, 40000)
    v = level + RNG.uniform(-0.004, 0.010)*level*t \
        + 0.18*level*np.sin(2*np.pi*(t % 12)/12 + RNG.uniform(0, 6.28)) \
        + RNG.normal(0, 0.04*level, MONTHS)
    return pd.Series(np.clip(v, 1, None), pd.date_range("2017-01-01", periods=MONTHS, freq="MS"))

def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))

series = [make_series() for _ in range(N)]
splits = [(s.iloc[:-H], s.iloc[-H:]) for s in series]

def run(fn, warm=True):
    if warm:
        try: fn(splits[0][0], H)
        except Exception: pass
    errs, fails, t0 = [], 0, time.perf_counter()
    for tr, te in splits:
        try: errs.append(rmse(fn(tr, H), te.values))
        except Exception: fails += 1; errs.append(np.nan)
    return np.nanmean(errs), time.perf_counter()-t0, fails

rows = []
rows.append(("seasonal-naive", "-", *run(naive_forecast, warm=False)))

from statsmodels.tsa.statespace.sarimax import SARIMAX
def sarima(tr, n):
    r = SARIMAX(tr, order=(1,1,1), seasonal_order=(1,1,0,12), enforce_stationarity=False,
                enforce_invertibility=False).fit(disp=False, maxiter=200)
    return r.get_forecast(n).predicted_mean.values
rows.append(("SARIMA (classical)", "-", *run(sarima, warm=False)))

from chronos import BaseChronosPipeline
CANDIDATES = [
    ("amazon/chronos-2",            "120M"),
    ("amazon/chronos-bolt-base",    "205M"),
    ("amazon/chronos-bolt-small",   "48M"),
    ("amazon/chronos-t5-small",     "46M"),
]
for mid, size in CANDIDATES:
    try:
        t0 = time.perf_counter()
        pipe = BaseChronosPipeline.from_pretrained(mid, device_map="cpu")
        load = time.perf_counter()-t0
        def mk(p):
            def f(tr, n):
                q, _ = p.predict_quantiles([torch.tensor(tr.values, dtype=torch.float32)],
                                           prediction_length=n, quantile_levels=[0.1,0.5,0.9])
                a = np.asarray(q[0], dtype=float)
                a = a[0] if a.ndim == 3 else a           # (V,H,Q) -> (H,Q)
                return a[:, 1]
            return f
        e, dt, f = run(mk(pipe))
        rows.append((mid.split("/")[-1], size, e, dt, f))
        print(f"  {mid:32s} loaded {load:5.1f}s  rmse {e:9.1f}  {dt/N*1000:6.1f} ms/series  fails {f}", flush=True)
    except Exception as ex:
        print(f"  {mid:32s} UNAVAILABLE: {type(ex).__name__}: {str(ex)[:70]}", flush=True)

base = rows[0][2]
print(f"\n{'model':26s} {'params':>7s} {'mean RMSE':>11s} {'vs naive':>9s} {'ms/series':>10s} {'fail':>5s}")
print("-"*76)
for name, size, e, dt, f in sorted(rows, key=lambda r: (np.isnan(r[2]), r[2])):
    print(f"{name:26s} {size:>7s} {e:11.1f} {e/base:8.2f}x {dt/N*1000:10.1f} {f:5d}")
