"""SARIMA vs Chronos-2 through the full Nexus pipeline on REAL data.

Mirrors notebook sections 3, 4, 6, 8 and 10 outside Jupyter:
  base forecast per region -> GAT-weighted blending -> random control -> tests.

    python scripts/run_comparison.py                 # electricity + carbon
    python scripts/run_comparison.py --resources electricity
    python scripts/run_comparison.py --limit-zones 20    # quick smoke test
"""
from __future__ import annotations
import argparse, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from nexus_local import chronos_forecast, load_chronos

HOLDOUT, ALPHA_MAX, MIN_ZONE_MONTHS, SEED = 12, 0.5, 36, 42
EMBER = os.path.join(ROOT, "data", "ember")


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ------------------------------------------------------------------ ember load
def _norm(s):
    return (str(s).strip().lower().replace("&", "and")
            .replace(".", "").replace(",", "").replace("  ", " "))


def load_ember_region(path, region):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in ["State", "State code", "Area", "ISO 3 code", "Area type",
                           "Date", "Category", "Variable", "Unit", "Value"] if c in cols]
    df = pd.read_csv(path, usecols=usecols)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    gen = df[(df["Category"] == "Electricity generation")
             & (df["Variable"].str.lower() == "total generation")
             & (df["Unit"].isin(["GWh", "TWh"]))].copy()
    gen["electricity_gwh"] = np.where(gen["Unit"] == "TWh", gen["Value"] * 1000.0, gen["Value"])
    car = df[(df["Category"] == "Power sector emissions")
             & (df["Variable"] == "CO2 intensity")].copy()
    car["co2_intensity"] = car["Value"]
    if region in ("US", "India"):
        iso = "USA" if region == "US" else "IND"
        for g in (gen, car):
            g["zone_id"] = iso + "|" + g["State"].map(_norm)
            g.drop(g.index[g["State"].astype(str).str.lower().str.contains("total")], inplace=True)
    else:
        gen = gen[gen["Area type"] == "Country or economy"]
        car = car[car["Area type"] == "Country or economy"]
        gen["zone_id"] = gen["ISO 3 code"].astype(str)
        car["zone_id"] = car["ISO 3 code"].astype(str)
    gen = gen.dropna(subset=["Value", "zone_id", "Date"])
    car = car.dropna(subset=["Value", "zone_id", "Date"])
    g = gen.groupby(["zone_id", "Date"], as_index=False)["electricity_gwh"].mean()
    c = car.groupby(["zone_id", "Date"], as_index=False)["co2_intensity"].mean()
    return pd.merge(g, c, on=["zone_id", "Date"], how="outer")


def build_region_series(zones_with_nodes):
    ember = pd.concat([
        load_ember_region(f"{EMBER}/us_monthly_full_release_long_format.csv", "US"),
        load_ember_region(f"{EMBER}/europe_monthly_full_release_long_format.csv", "EU"),
        load_ember_region(f"{EMBER}/india_monthly_full_release_long_format.csv", "India"),
    ], ignore_index=True)
    out = {"electricity": {}, "carbon": {}}
    for z in sorted(zones_with_nodes):
        sub = ember[ember.zone_id == z].set_index("Date").sort_index()
        if sub.empty:
            continue
        for rk, col in [("electricity", "electricity_gwh"), ("carbon", "co2_intensity")]:
            s = sub[col].dropna()
            s = s[~s.index.duplicated(keep="first")]
            if len(s) >= MIN_ZONE_MONTHS:
                out[rk][z] = s
    return out


# ------------------------------------------------------------------- water load
def build_basin_map(nodes_df):
    """Nearest-centroid basin assignment, using the notebook's own
    BASIN_CENTROIDS / _haversine_km so the mapping is identical."""
    import json as _json
    nb = _json.load(open(os.path.join(ROOT, "notebooks",
                                      "gat_weighted_forecasting.ipynb"), encoding="utf-8"))
    ns = {"np": np, "pd": pd}
    exec("".join(nb["cells"][3]["source"]), ns)
    BC, MAXKM, hav = ns["BASIN_CENTROIDS"], ns["BASIN_MAX_KM"], ns["_haversine_km"]
    tws = pd.read_csv(os.path.join(ROOT, "data", "g3p", "G3P_v1.12_tws_rivbas.csv"),
                      parse_dates=["time [yyyy-mm-dd]"])
    tws = tws.rename(columns={"time [yyyy-mm-dd]": "date"}).set_index("date").sort_index()
    basins = [c.replace(" [mm]", "") for c in tws.columns
              if c.endswith("[mm]") and not c.startswith("uncertainty") and c != "global [mm]"]
    cand = [b for b in basins if b in BC]
    clat = np.array([BC[b][0] for b in cand]); clon = np.array([BC[b][1] for b in cand])
    out = {}
    for sid, la, lo in zip(nodes_df.site_id.astype(int), nodes_df.latitude, nodes_df.longitude):
        d = hav(float(la), float(lo), clat, clon); j = int(np.argmin(d))
        out[sid] = cand[j] if d[j] <= MAXKM else "Unknown"
    return out, tws


def build_water_series(basin_of_site, tws):
    have = {b for b in basin_of_site.values() if b != "Unknown"}
    out = {}
    for b in sorted(have):
        s = tws[f"{b} [mm]"].dropna()
        s = s[~s.index.duplicated(keep="first")]
        if len(s) >= MIN_ZONE_MONTHS:
            out[b] = s
    return out


# ---------------------------------------------------------------- weight struct
def build_weight_struct(path, label):
    """Notebook §4: min-max scale attention to [0, ALPHA_MAX], drop self-loops."""
    W = torch.load(path, map_location="cpu", weights_only=False)
    site_id = W["site_id"].numpy().astype(int)
    att = W["site_attention_received"].numpy().astype(float)
    ei = W["attention_edge_index"].numpy()
    ew = W["attention_edge_weight"].numpy().astype(float)
    keep = ei[0] != ei[1]
    ei, ew = ei[:, keep], ew[keep]
    rng = att.max() - att.min()
    a = ALPHA_MAX * ((att - att.min()) / rng if rng > 0 else np.zeros_like(att))
    n_nbr = np.bincount(ei[1], minlength=len(site_id))
    print(f"  {label:16s} N={len(site_id)} edges={ei.shape[1]} "
          f"a[min/mean/max]={a.min():.3f}/{a.mean():.3f}/{a.max():.3f} "
          f"nodes_with_nbr={(n_nbr > 0).sum()}")
    return {"site_id": site_id, "a": a, "src": ei[0], "dst": ei[1], "w": ew}


def blend(node_base, valid, src, dst, w, a_vec):
    """Notebook §8: (1-a)*own + a*attention-weighted neighbour mean."""
    N, H = node_base.shape
    num, den = np.zeros((N, H)), np.zeros(N)
    good = valid[src]
    se, de, we = src[good], dst[good], w[good]
    np.add.at(num, de, we[:, None] * node_base[se])
    np.add.at(den, de, we)
    pos = den[:, None] > 0
    nbr = np.where(pos, num / np.where(pos, den[:, None], 1.0), node_base)
    a_eff = np.where(den > 0, a_vec, 0.0)
    return (1 - a_eff)[:, None] * node_base + a_eff[:, None] * nbr, (den > 0)


# ------------------------------------------------------------------ forecasters
def sarima_forecast(train, n):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    r = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12),
                enforce_stationarity=False, enforce_invertibility=False
                ).fit(disp=False, maxiter=200)
    return r.get_forecast(n).predicted_mean.values


MODELS = {"SARIMA": sarima_forecast, "Chronos-2": chronos_forecast}


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def mape(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.abs(b) > 1e-8
    return float(np.mean(np.abs((b[m] - a[m]) / b[m])) * 100) if m.any() else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resources", default="electricity,carbon")
    ap.add_argument("--limit-zones", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs"))
    args = ap.parse_args()
    resources = [r.strip() for r in args.resources.split(",") if r.strip()]
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_parquet(os.path.join(ROOT, "data", "aligned_dataset.parquet"))
    nodes = df[df.is_node]
    zone_of_site = dict(zip(nodes.site_id.astype(int), nodes.grid_zone_id))
    zones_with_nodes = {z for z in nodes.grid_zone_id.unique() if isinstance(z, str)}

    section("Region series")
    t0 = time.perf_counter()
    region_series = build_region_series(zones_with_nodes) if (
        "electricity" in resources or "carbon" in resources) else {}
    REGION_OF = {}
    sid_order = None
    if "water" in resources:
        basin_of_site, tws = build_basin_map(nodes)
        region_series["water"] = build_water_series(basin_of_site, tws)
    for r in resources:
        print(f"  {r:12s}: {len(region_series[r])} zones")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    _W = torch.load(os.path.join(ROOT, "data", "weights", "gat_weights_colocation.pt"),
                    map_location="cpu", weights_only=False)
    _sid = _W["site_id"].numpy().astype(int)
    for r in resources:
        m = basin_of_site if r == "water" else zone_of_site
        REGION_OF[r] = np.array([m.get(int(s)) for s in _sid], dtype=object)

    if args.limit_zones:
        for r in resources:
            keep = sorted(region_series[r])[:args.limit_zones]
            region_series[r] = {k: region_series[r][k] for k in keep}
        print(f"  (limited to {args.limit_zones} zones per resource)")

    section("Graph structures (real .pt payloads)")
    structs = {
        "gat_weighted": build_weight_struct(
            os.path.join(ROOT, "data", "weights", "gat_weights_colocation.pt"), "colocation"),
        "random_control": build_weight_struct(
            os.path.join(ROOT, "data", "weights", "gat_weights_random_control.pt"), "random"),
    }

    globals()["REGION_OF"] = REGION_OF
    section("Pass 1 - base forecasts per region")
    load_chronos()
    base, actuals, rows1 = {}, {}, []
    for res in resources:
        actuals[res] = {}
        for name, fn in MODELS.items():
            base[(name, res)] = {}
            t0, fails = time.perf_counter(), 0
            for z, s in region_series[res].items():
                tr, te = s.iloc[:-HOLDOUT], s.iloc[-HOLDOUT:]
                actuals[res][z] = te.values
                try:
                    base[(name, res)][z] = np.asarray(fn(tr, HOLDOUT), float)
                except Exception:
                    fails += 1
            dt = time.perf_counter() - t0
            e = [rmse(base[(name, res)][z], actuals[res][z]) for z in base[(name, res)]]
            p = [mape(base[(name, res)][z], actuals[res][z]) for z in base[(name, res)]]
            rows1.append({"resource": res, "model": name, "zones": len(base[(name, res)]),
                          "RMSE": np.mean(e), "MAPE": np.nanmean(p), "secs": dt, "fails": fails})
            print(f"  {res:12s} {name:10s} {len(base[(name,res)]):3d} zones  "
                  f"RMSE {np.mean(e):10.1f}  MAPE {np.nanmean(p):6.2f}%  {dt:6.1f}s  fails {fails}")

    pd.DataFrame(rows1).to_csv(os.path.join(args.out, "compare_base.csv"), index=False)

    section("Pass 1 verdict - Chronos-2 vs SARIMA (per zone, paired)")
    for res in resources:
        zs = sorted(set(base[("SARIMA", res)]) & set(base[("Chronos-2", res)]))
        a = np.array([rmse(base[("Chronos-2", res)][z], actuals[res][z]) for z in zs])
        b = np.array([rmse(base[("SARIMA", res)][z], actuals[res][z]) for z in zs])
        t_s, t_p = stats.ttest_rel(a, b)
        try:
            w_s, w_p = stats.wilcoxon(a, b)
        except Exception:
            w_p = np.nan
        win = int((a < b).sum())
        better = "Chronos-2 better" if a.mean() < b.mean() else "SARIMA better"
        sig = "SIGNIFICANT" if t_p < 0.05 else "not significant"
        print(f"  {res:12s} n={len(zs):3d}  chronos={a.mean():10.1f}  sarima={b.mean():10.1f}  "
              f"chronos wins {win}/{len(zs)}  t_p={t_p:.3g}  w_p={w_p:.3g}  -> {better}, {sig}")

    section("Pass 2 - per-site GAT blending vs random control")
    rows = []
    for res in resources:
        for model in MODELS:
            bz = base[(model, res)]
            for cond, st in structs.items():
                N = len(st["site_id"])
                nb_ = np.full((N, HOLDOUT), np.nan)
                act = np.full((N, HOLDOUT), np.nan)
                for i, z in enumerate(REGION_OF[res]):
                    if z in bz:
                        nb_[i] = bz[z]
                    if z in actuals[res]:
                        act[i] = actuals[res][z]
                valid = ~np.isnan(nb_).any(axis=1)
                blended, had = blend(np.nan_to_num(nb_), valid, st["src"], st["dst"], st["w"], st["a"])
                m = valid & ~np.isnan(act).any(axis=1)
                for i in np.where(m)[0]:
                    rows.append({"resource": res, "model": model, "condition": cond,
                                 "site": int(st["site_id"][i]),
                                 "rmse_unw": rmse(nb_[i], act[i]),
                                 "rmse_blend": rmse(blended[i], act[i]),
                                 "changed": bool(had[i])})
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(args.out, "compare_sites.csv"), index=False)

    print(f"{'resource':12s} {'model':10s} {'condition':15s} {'sites':>6s} {'chg':>6s} "
          f"{'unweighted':>11s} {'blended':>10s} {'delta':>9s}")
    print("-" * 88)
    for (res, mo, co), g in R.groupby(["resource", "model", "condition"]):
        d = g.rmse_blend.mean() - g.rmse_unw.mean()
        print(f"{res:12s} {mo:10s} {co:15s} {len(g):6d} {g.changed.sum():6d} "
              f"{g.rmse_unw.mean():11.1f} {g.rmse_blend.mean():10.1f} {d:+9.2f}")

    section("Pass 2 verdict - does the co-location graph beat random edges?")
    for (res, mo), g in R.groupby(["resource", "model"]):
        gat = g[g.condition == "gat_weighted"].set_index("site")["rmse_blend"]
        rnd = g[g.condition == "random_control"].set_index("site")["rmse_blend"]
        unw = g[g.condition == "gat_weighted"].set_index("site")["rmse_unw"]
        common = gat.index.intersection(rnd.index)
        a, b, u = gat.loc[common].values, rnd.loc[common].values, unw.loc[common].values
        for other, ov in [("random", b), ("unweighted", u)]:
            if len(common) < 3 or np.allclose(a, ov):
                print(f"  {res:12s} {mo:10s} gat vs {other:11s} identical / n<3 -> no test")
                continue
            t_s, t_p = stats.ttest_rel(a, ov)
            verdict = ("GAT lower RMSE" if a.mean() < ov.mean() else "GAT higher RMSE") \
                      if t_p < 0.05 else "no significant difference"
            print(f"  {res:12s} {mo:10s} gat vs {other:11s} n={len(common):5d}  "
                  f"gat={a.mean():9.2f}  {other}={ov.mean():9.2f}  "
                  f"diff={a.mean()-ov.mean():+8.3f}  t_p={t_p:.3g}  -> {verdict}")

    print(f"\nwrote {args.out}/compare_base.csv and compare_sites.csv")


if __name__ == "__main__":
    main()
