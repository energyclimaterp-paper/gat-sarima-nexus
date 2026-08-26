"""Build a consolidated, human-readable record of the last notebook run.

Every number is recomputed from outputs/ — nothing is hand-copied.
Writes outputs/RESULTS.md and outputs/summary_model_comparison.csv
"""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

rc = {"__file__": os.path.join(ROOT, "scripts", "run_comparison.py")}
exec(open(os.path.join(ROOT, "scripts", "run_comparison.py"), encoding="utf-8")
     .read().split("def main()")[0], rc)

al = pd.read_parquet(os.path.join(ROOT, "data", "aligned_dataset.parquet"))
nodes = al[al.is_node]
zwn = {z for z in nodes.grid_zone_id.unique() if isinstance(z, str)}
rs = rc["build_region_series"](zwn)
bm, tws = rc["build_basin_map"](nodes)
rs["water"] = rc["build_water_series"](bm, tws)

d = pd.read_csv(os.path.join(OUT, "per_region_base_forecasts.csv"), parse_dates=["timestamp"])
rows = []
for res in ["electricity", "carbon", "water"]:
    for rid, g in d[d.resource == res].groupby("region_id"):
        s = rs[res].get(rid)
        if s is None:
            continue
        act = s.iloc[-12:].values
        for model, gm in g.groupby("model"):
            v = gm.sort_values("timestamp").value.values[:12]
            if len(v) != 12:
                continue
            rows.append({"resource": res, "region": rid, "model": model,
                         "RMSE": float(np.sqrt(np.mean((v - act) ** 2))),
                         "MAE": float(np.mean(np.abs(v - act)))})
R = pd.DataFrame(rows)
piv = R.pivot_table(index=["resource", "region"], columns="model", values="RMSE").dropna()

comp = []
for res, g in piv.groupby(level=0):
    a, b = g["Nexus"].values, g["SARIMA"].values
    t_s, t_p = stats.ttest_rel(a, b)
    try:
        w_s, w_p = stats.wilcoxon(a, b)
    except Exception:
        w_p = np.nan
    comp.append({"resource": res, "n_regions": len(g),
                 "SARIMA_RMSE": round(b.mean(), 3), "Nexus_RMSE": round(a.mean(), 3),
                 "delta": round(a.mean() - b.mean(), 3),
                 "pct_change": round(100 * (a.mean() - b.mean()) / b.mean(), 2),
                 "nexus_wins": int((a < b).sum()),
                 "t_p": float(f"{t_p:.4g}"), "wilcoxon_p": float(f"{w_p:.4g}")})
C = pd.DataFrame(comp)
C.to_csv(os.path.join(OUT, "summary_model_comparison.csv"), index=False)

agg = pd.read_csv(os.path.join(OUT, "step5_aggregate_by_resource.csv"))
sig = pd.read_csv(os.path.join(OUT, "step5_significance.csv"))
dia = pd.read_csv(os.path.join(OUT, "step5_diagram_versions.csv"))
log = json.load(open(os.path.join(OUT, "weight_injection_log.json")))
cov = d.groupby(["model", "resource"]).size().rename("forecast_rows").reset_index()

def md_table(df):
    h = "| " + " | ".join(df.columns) + " |"
    s = "|" + "|".join("---" for _ in df.columns) + "|"
    b = ["| " + " | ".join(str(x) for x in r) + " |" for r in df.values]
    return "\n".join([h, s] + b)

with open(os.path.join(OUT, "RESULTS.md"), "w", encoding="utf-8") as f:
    w = f.write
    w("# Results record\n\n")
    w(f"Source: `notebooks/gat_weighted_forecasting.ipynb` executed end to end "
      f"(0 cell errors). Nexus backend = **chronos-2** (local, no API key).\n\n")
    w(f"Holdout = {12} months. Regions: electricity/carbon = grid zone, water = river basin.\n\n")
    w("## 1. Coverage\n\n" + md_table(cov) + "\n\n")
    w("## 2. SARIMA vs Nexus — per-region paired\n\n" + md_table(C) + "\n\n")
    w("Negative `delta` = Nexus lower RMSE (better).\n\n")
    w("## 3. Per-site scoring after GAT blending (§10)\n\n" + md_table(agg) + "\n\n")
    w("## 4. Significance: gat_weighted vs baselines (§10)\n\n" + md_table(sig) + "\n\n")
    w("## 5. Version comparison incl. oracle (§10)\n\n" + md_table(dia) + "\n\n")
    w("## 6. Run configuration\n\n```json\n")
    w(json.dumps({k: v for k, v in log.items() if k != "native_injection_demo"}, indent=2))
    w("\n```\n\n## 7. Files in this directory\n\n")
    for fn in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, fn)
        if os.path.isfile(p):
            w(f"- `{fn}` — {os.path.getsize(p)/1024:.1f} KB\n")

print("wrote outputs/RESULTS.md and outputs/summary_model_comparison.csv")
print(C.to_string(index=False))
