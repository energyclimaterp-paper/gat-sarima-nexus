"""MAE, WMAPE, RMSE, Bias and MASE for the three blending conditions.

Conditions (one file each, written by Part B Section 9):
  unweighted      -- simple average across models, no graph
  gat_weighted    -- attention-weighted blending over the co-location graph
  random_control  -- attention-weighted blending over the random-edge graph

Method
------
Part B's own step5b aggregates *site-level error metrics* to region level. That
cannot produce WMAPE / Bias / MASE, which need signed forecast-actual pairs. So
this recomputes from the raw site-level forecasts:

  1. ensemble across models  (mean per site x resource x timestamp) -- the same
     ensembling step the notebook's `ensemble_score` performs
  2. map site -> region      (grid zone for electricity/carbon, basin for water)
  3. region prediction       = mean over the sites in that region
  4. score against the region's held-out final 12 months

MASE uses the seasonal naive denominator with m = 12, computed on the TRAINING
portion only (everything before the 12-month holdout), so the scale factor never
sees the evaluation window.
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs_v2", "partB")
HOLDOUT, SEASON = 12, 12

CONDITIONS = {
    "unweighted": "forecasts_unweighted.csv",
    "gat_weighted": "forecasts_gat_weighted.csv",
    "random_control": "forecasts_random_control_weighted.csv",
}


def region_maps():
    """site_id -> region, per resource. Mirrors REGION_OF in Part B."""
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    ns = {"__file__": os.path.join(ROOT, "scripts", "run_comparison.py")}
    exec(open(ns["__file__"], encoding="utf-8").read().split("def main()")[0], ns)
    al = pd.read_parquet(os.path.join(ROOT, "data", "aligned_dataset.parquet"))
    nodes = al[al.is_node]
    zone = dict(zip(nodes.site_id.astype(int), nodes.grid_zone_id))
    basin, _ = ns["build_basin_map"](nodes)
    return {"electricity": zone, "carbon": zone, "water": basin}


def load_series():
    rs = json.load(open(os.path.join(OUT, "region_series_cache.json")))
    out = {}
    for res, d in rs.items():
        out[res] = {}
        for rid, ser in d.items():
            v = ser["values"] if isinstance(ser, dict) and "values" in ser else ser
            out[res][rid] = np.asarray(v, dtype=float)
    return out


def mase_denominator(train, m=SEASON):
    """In-sample seasonal naive MAE. Falls back to the lag-1 naive if the series
    is too short for a full seasonal lag."""
    if len(train) > m:
        d = np.abs(train[m:] - train[:-m])
    elif len(train) > 1:
        d = np.abs(np.diff(train))
    else:
        return np.nan
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) and d.mean() > 0 else np.nan


def metrics(y, f, denom):
    e = f - y
    ae = np.abs(e)
    return {
        "MAE": float(ae.mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "WMAPE_%": float(100 * ae.sum() / np.abs(y).sum()) if np.abs(y).sum() > 0 else np.nan,
        "Bias": float(e.mean()),
        "Bias_%": float(100 * e.sum() / np.abs(y).sum()) if np.abs(y).sum() > 0 else np.nan,
        "MASE": float(ae.mean() / denom) if denom and np.isfinite(denom) else np.nan,
    }


def main():
    REG = region_maps()
    SERIES = load_series()
    rows = []

    for cond, fn in CONDITIONS.items():
        df = pd.read_csv(os.path.join(OUT, fn), parse_dates=["timestamp"])
        # 1. ensemble across models
        ens = df.groupby(["site_id", "resource", "timestamp"], as_index=False)["value"].mean()
        # 2. site -> region
        ens["region_id"] = [REG[r].get(int(s)) for s, r in zip(ens.site_id, ens.resource)]
        ens = ens.dropna(subset=["region_id"])
        # 3. region prediction
        reg = ens.groupby(["resource", "region_id", "timestamp"], as_index=False)["value"].mean()

        for res, g in reg.groupby("resource"):
            per_region = []
            for rid, gg in g.groupby("region_id"):
                s = SERIES.get(res, {}).get(rid)
                if s is None or len(s) < HOLDOUT + 1:
                    continue
                y = s[-HOLDOUT:]
                f = gg.sort_values("timestamp").value.values[:HOLDOUT]
                if len(f) != HOLDOUT or not np.isfinite(f).all():
                    continue
                per_region.append(metrics(y, f, mase_denominator(s[:-HOLDOUT])))
            if not per_region:
                continue
            M = pd.DataFrame(per_region)
            rows.append({"resource": res, "condition": cond, "N_regions": len(M),
                         **{k: round(float(M[k].mean()), 4) for k in M.columns}})

    # ---- second basis: score each SITE, then average within region, then over regions.
    # This is Part B step5b's basis. It differs from the above for blended conditions
    # because mean(RMSE_site) >= RMSE(mean prediction) by Jensen; identical for
    # `unweighted`, where every site in a region carries the same forecast.
    rows_site = []
    for cond, fn in CONDITIONS.items():
        df = pd.read_csv(os.path.join(OUT, fn), parse_dates=["timestamp"])
        ens = df.groupby(["site_id", "resource", "timestamp"], as_index=False)["value"].mean()
        ens["region_id"] = [REG[r].get(int(s)) for s, r in zip(ens.site_id, ens.resource)]
        ens = ens.dropna(subset=["region_id"])
        for res, g in ens.groupby("resource"):
            per_site = []
            for (sid, rid), gg in g.groupby(["site_id", "region_id"]):
                s_ = SERIES.get(res, {}).get(rid)
                if s_ is None or len(s_) < HOLDOUT + 1:
                    continue
                y = s_[-HOLDOUT:]
                f = gg.sort_values("timestamp").value.values[:HOLDOUT]
                if len(f) != HOLDOUT or not np.isfinite(f).all():
                    continue
                m = metrics(y, f, mase_denominator(s_[:-HOLDOUT]))
                m["region_id"] = rid
                per_site.append(m)
            if not per_site:
                continue
            S = pd.DataFrame(per_site)
            byreg = S.groupby("region_id").mean(numeric_only=True)
            rows_site.append({"resource": res, "condition": cond, "N_regions": len(byreg),
                              **{k: round(float(byreg[k].mean()), 4) for k in byreg.columns}})
    # ---- pooled ("micro") basis for the scale-free metrics.
    # WMAPE and Bias% averaged ACROSS regions are macro-averages: every region counts
    # equally regardless of size, so a 36 GWh zone moves the number as much as a
    # 58,608 GWh one. That is why electricity gat_weighted reports Bias = -177 (an
    # under-forecast in absolute terms) alongside Bias_% = +4.6. WMAPE is volume-
    # weighted by definition, so the pooled form below is the one to report.
    rows_pool = []
    for cond, fn in CONDITIONS.items():
        df = pd.read_csv(os.path.join(OUT, fn), parse_dates=["timestamp"])
        ens = df.groupby(["site_id", "resource", "timestamp"], as_index=False)["value"].mean()
        ens["region_id"] = [REG[r].get(int(s)) for s, r in zip(ens.site_id, ens.resource)]
        ens = ens.dropna(subset=["region_id"])
        reg = ens.groupby(["resource", "region_id", "timestamp"], as_index=False)["value"].mean()
        for res, g in reg.groupby("resource"):
            Y, F, scales = [], [], []
            for rid, gg in g.groupby("region_id"):
                s_ = SERIES.get(res, {}).get(rid)
                if s_ is None or len(s_) < HOLDOUT + 1:
                    continue
                f = gg.sort_values("timestamp").value.values[:HOLDOUT]
                if len(f) != HOLDOUT or not np.isfinite(f).all():
                    continue
                Y.append(s_[-HOLDOUT:]); F.append(f)
                scales.append(mase_denominator(s_[:-HOLDOUT]))
            if not Y:
                continue
            y = np.concatenate(Y); f = np.concatenate(F)
            e = f - y
            sc = np.repeat(np.asarray(scales, float), HOLDOUT)
            ok = np.isfinite(sc) & (sc > 0)
            rows_pool.append({
                "resource": res, "condition": cond, "N_regions": len(Y),
                "MAE": round(float(np.abs(e).mean()), 4),
                "WMAPE_%": round(float(100 * np.abs(e).sum() / np.abs(y).sum()), 4),
                "RMSE": round(float(np.sqrt((e ** 2).mean())), 4),
                "Bias": round(float(e.mean()), 4),
                "Bias_%": round(float(100 * e.sum() / np.abs(y).sum()), 4),
                "MASE": round(float((np.abs(e)[ok] / sc[ok]).mean()), 4),
            })

    order = ["resource", "condition", "N_regions", "MAE", "WMAPE_%", "RMSE", "Bias", "Bias_%", "MASE"]

    RP = pd.DataFrame(rows_pool).sort_values(["resource", "condition"])[order]
    RP.to_csv(os.path.join(OUT, "blending_metrics_pooled.csv"), index=False)
    print("#" * 72)
    print("BASIS C -- POOLED across regions (recommended for WMAPE / Bias%)")
    print("#" * 72)
    for res in ["electricity", "carbon", "water"]:
        sub = RP[RP.resource == res]
        if sub.empty:
            continue
        print(f"\n=== {res.upper()}  (N = {int(sub.N_regions.iloc[0])} regions)")
        print(sub.drop(columns=["resource", "N_regions"]).to_string(index=False))

    RS = pd.DataFrame(rows_site).sort_values(["resource", "condition"])
    RS[order].to_csv(os.path.join(OUT, "blending_metrics_sitebasis.csv"), index=False)
    print("\n" + "#" * 72)
    print("BASIS B -- score per site, average within region, average over regions")
    print("(matches Part B step5b_region_metrics)")
    print("#" * 72)
    for res in ["electricity", "carbon", "water"]:
        sub = RS[RS.resource == res]
        if sub.empty:
            continue
        print(f"\n=== {res.upper()}  (N = {int(sub.N_regions.iloc[0])} regions)")
        print(sub[order].drop(columns=["resource", "N_regions"]).to_string(index=False))

    R = pd.DataFrame(rows).sort_values(["resource", "condition"])
    R = R[order]
    R.to_csv(os.path.join(OUT, "blending_metrics.csv"), index=False)

    for res in ["electricity", "carbon", "water"]:
        sub = R[R.resource == res]
        if sub.empty:
            continue
        print(f"\n=== {res.upper()}  (N = {int(sub.N_regions.iloc[0])} regions)")
        print(sub.drop(columns=["resource", "N_regions"]).to_string(index=False))
    print(f"\nwrote {os.path.join(OUT, 'blending_metrics.csv')}")
    print("\nMASE < 1 beats the in-sample seasonal naive; Bias > 0 = over-forecast.")


if __name__ == "__main__":
    main()
