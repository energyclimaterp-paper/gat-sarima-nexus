"""PARTS 1-4, 6 -- audit, Nexus provenance, seasonal-naive baseline, conditions.

Uses cached base forecasts; no forecaster is rerun.
"""
from __future__ import annotations
import hashlib, json, os, platform, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTA = os.path.join(ROOT, "outputs_v2", "partA")
PARTB = os.path.join(ROOT, "outputs_v2", "partB")
AUDIT = os.path.join(ROOT, "outputs_v2", "audit")
os.makedirs(AUDIT, exist_ok=True)
HOLDOUT, SEASON, ALPHA_MAX = 12, 12, 0.5
R = {}
T0 = time.time()


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# =============================================================== PART 1
def part1(series, nodes, zone, basin):
    hdr("1.2 REGION-COUNT PROVENANCE  (94 pre-fix vs 95 post-fix)")
    pre_p = os.path.join(ROOT, "archive_v1", "aligned_dataset.parquet")
    cur = pd.read_parquet(os.path.join(ROOT, "data", "aligned_dataset.parquet"))
    if os.path.exists(pre_p):
        pre = pd.read_parquet(pre_p)
        pz = {z for z in pre.loc[pre.is_node, "grid_zone_id"].unique() if isinstance(z, str)}
        cz = {z for z in cur.loc[cur.is_node, "grid_zone_id"].unique() if isinstance(z, str)}
        print(f"  pre-fix checkpoint EXISTS: {pre_p}")
        print(f"  pre-fix  is_node={int(pre.is_node.sum())}  distinct node zones={len(pz)}")
        print(f"  post-fix is_node={int(cur.is_node.sum())}  distinct node zones={len(cz)}")
        print(f"  zones added  : {sorted(cz - pz)}")
        print(f"  zones removed: {sorted(pz - cz)}")
        newsites = set(cur.loc[cur.is_node, "site_id"]) - set(pre.loc[pre.is_node, "site_id"])
        print(f"  node sites added: {len(newsites)} -> {sorted(newsites)[:10]}")
        if newsites:
            sub = cur[cur.site_id.isin(newsites)][["site_id", "country", "grid_zone_id", "temperature_c"]]
            print(sub.to_string(index=False))
        pre_t = pre.temperature_c.notna().sum(); cur_t = cur.temperature_c.notna().sum()
        print(f"\n  temperature_c non-null: pre-fix={pre_t}  post-fix={cur_t}  (delta={cur_t-pre_t})")
        cause = ("live Open-Meteo refetch changed temperature coverage, which changes is_node "
                 "(is_node requires all 4 node features present)") if cur_t != pre_t else \
                "temperature coverage unchanged -- cause lies elsewhere"
        print(f"  ATTRIBUTED CAUSE: {cause}")
        R["1.2_region_provenance"] = {"pre_is_node": int(pre.is_node.sum()),
                                      "post_is_node": int(cur.is_node.sum()),
                                      "zones_added": sorted(cz - pz), "zones_removed": sorted(pz - cz),
                                      "sites_added": sorted(int(x) for x in newsites),
                                      "pre_temp_nonnull": int(pre_t), "post_temp_nonnull": int(cur_t),
                                      "cause": cause}
    else:
        print("  no pre-fix checkpoint -- cannot attribute")

    print(f"\n  MIN_ZONE_MONTHS=36  HOLDOUT={HOLDOUT}")
    for res in ["electricity", "carbon", "water"]:
        L = {k: len(v) for k, v in series.get(res, {}).items()}
        if L:
            print(f"    {res:12s}: {len(L)} regions pass, months min={min(L.values())} "
                  f"median={int(np.median(list(L.values())))} max={max(L.values())}")

    hdr("1.3 LEAKAGE-FIX CONFIRMATION  (node-feature means, pre vs post)")
    if os.path.exists(pre_p):
        pre = pd.read_parquet(pre_p)
        rows = []
        for col in ["electricity_gwh", "co2_intensity", "water_stress_bws", "temperature_c"]:
            a = pre.loc[pre.is_node, col]; b = cur.loc[cur.is_node, col]
            a = a[(a != -9999) & a.notna()]; b = b[(b != -9999) & b.notna()]
            rows.append({"feature": col, "pre_mean": round(float(a.mean()), 6),
                         "post_mean": round(float(b.mean()), 6),
                         "differs": bool(abs(a.mean() - b.mean()) > 1e-9)})
        T = pd.DataFrame(rows)
        print(T.to_string(index=False))
        ec = T[T.feature.isin(["electricity_gwh", "co2_intensity"])]
        if ec.differs.all():
            print("  ASSERTION HELD: electricity_gwh and co2_intensity means DIFFER from the pre-fix run.")
        else:
            same = list(ec[~ec.differs].feature)
            print(f"  *** WARNING: {same} did NOT change between pre-fix and post-fix artifacts.")
            print("      The recut cannot be confirmed from these two checkpoints for those features.")
        R["1.3_leakage"] = T.to_dict("records")
    else:
        print("  no pre-fix baseline exists; printing current means only (no comparison possible)")


# =============================================================== PART 2
def part2(base, series, actuals):
    hdr("2.1 / 2.2 NEXUS PROVENANCE")
    nbp = os.path.join(ROOT, "notebooks", "v2", "partB_executed.ipynb")
    counts = {"GEMINI_CALLS": None, "via_LLM": None, "via_stat_fallback": None, "total": None}
    if os.path.exists(nbp):
        nb = json.load(open(nbp, encoding="utf-8"))
        txt = ""
        for c in nb["cells"]:
            if c["cell_type"] == "code" and "def nexus_forecast" in "".join(c["source"]):
                for o in c.get("outputs", []):
                    if o.get("output_type") == "stream":
                        txt += "".join(o["text"])
        import re
        m = re.search(r"Gemini calls actually made:\s*(\d+)\s*\|\s*via LLM:\s*(\d+)\s*\|\s*"
                      r"via statistical fallback:\s*(\d+)", txt)
        if m:
            counts = {"GEMINI_CALLS": int(m.group(1)), "via_LLM": int(m.group(2)),
                      "via_stat_fallback": int(m.group(3)),
                      "total": int(m.group(2)) + int(m.group(3))}
        backend = "chronos" if "Nexus backend: chronos" in txt else ("gemini" if "Nexus LLM ready" in txt else "unknown")
    else:
        backend = "unknown"

    tot = counts["total"] or 0
    fb = counts["via_stat_fallback"] or 0
    print(f"  backend actually used      : {backend}")
    print(f"  'GEMINI_CALLS' counter     : {counts['GEMINI_CALLS']}")
    print(f"  series via model path      : {counts['via_LLM']}")
    print(f"  series via _nexus_statistical: {fb}  ({100*fb/max(tot,1):.1f}%)")
    print(f"  series that attempted LLM then fell back after retry/parse failure: {fb}")
    print("\n  NEXUS PROVENANCE BLOCK")
    if backend == "chronos":
        print("    The Nexus column is NEITHER an LLM forecaster NOR the statistical fallback.")
        print("    It is amazon/chronos-2, a time-series foundation model run locally. The")
        print("    'GEMINI_CALLS' counter is reused bookkeeping: 0 network calls were made to")
        print("    any LLM. _nexus_statistical (0.8*SARIMA + 0.2*climatology) was never invoked.")
        print("    *** Do not describe Nexus as an LLM or multi-agent forecaster in the paper. ***")
        print("    *** Describe it as: Chronos-2 occupying the Nexus slot.                    ***")
    if tot and 100 * fb / tot > 10:
        print(f"\n*** WARNING: the Nexus result is not attributable to the LLM for {fb}/{tot} series.")
        print("    Do not describe Nexus as an LLM or multi-agent forecaster in the paper")
        print("    without reporting this split. ***")
    counts["backend"] = backend
    R["2_nexus_provenance"] = counts

    # ---- 2.3 Nexus_statistical_only, run deliberately on every series
    hdr("2.3 Nexus_statistical_only  (0.8*SARIMA + 0.2*month-of-year climatology)")
    rows = []
    for res in ["electricity", "carbon", "water"]:
        for rid, y in actuals.get(res, {}).items():
            sar = base.get(("SARIMA", res), {}).get(rid)
            nx = base.get(("Nexus", res), {}).get(rid)
            s = series[res][rid][:-HOLDOUT]
            if sar is None or nx is None or len(s) < SEASON:
                continue
            # month-of-year climatology aligned to the holdout positions
            idx = np.arange(len(s))
            mm = np.array([np.nanmean(s[(idx % SEASON) == ((len(s) + k) % SEASON)])
                           for k in range(HOLDOUT)])
            mm = np.where(np.isfinite(mm), mm, np.nanmean(s))
            stat = 0.8 * np.asarray(sar, float) + 0.2 * mm
            rmse = lambda f: float(np.sqrt(np.mean((np.asarray(f) - y) ** 2)))
            rows.append({"resource": res, "region": rid,
                         "Nexus": rmse(nx), "Nexus_statistical_only": rmse(stat),
                         "SARIMA": rmse(sar)})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(AUDIT, "part2_3_nexus_statistical_only.csv"), index=False)
    out = []
    for res, g in D.groupby("resource"):
        gap = g.Nexus.mean() - g.Nexus_statistical_only.mean()
        try:
            _, p = stats.wilcoxon(g.Nexus, g.Nexus_statistical_only)
        except Exception:
            p = np.nan
        neg = "Nexus BETTER" if gap < 0 else "Nexus WORSE"
        print(f"  {res:12s} n={len(g):3d}  Nexus={g.Nexus.mean():9.3f}  "
              f"stat_only={g.Nexus_statistical_only.mean():9.3f}  gap={gap:+9.3f}  "
              f"p={p:.4g}  -> {neg}")
        out.append({"resource": res, "n": len(g), "Nexus_RMSE": round(float(g.Nexus.mean()), 4),
                    "Nexus_stat_only_RMSE": round(float(g.Nexus_statistical_only.mean()), 4),
                    "gap": round(float(gap), 4), "wilcoxon_p": float(p),
                    "verdict": neg})
    print("\n  The gap is the contribution of the non-statistical model occupying the Nexus slot.")
    R["2.3_nexus_statistical_only"] = out


# =============================================================== PART 3
def part3(base, series, actuals):
    hdr("3.1 / 3.2 / 3.3 SEASONAL-NAIVE BASELINE, SKILL SCORE, MASE")
    rows = []
    for res in ["electricity", "carbon", "water"]:
        for rid, y in actuals.get(res, {}).items():
            s = series[res][rid]
            tr = s[:-HOLDOUT]
            if len(tr) < SEASON + 2:
                continue
            sn = tr[-SEASON:][:HOLDOUT]                      # yhat[t] = y[t-12]
            if len(sn) < HOLDOUT:
                continue
            d = np.abs(tr[SEASON:] - tr[:-SEASON])
            d = d[np.isfinite(d)]
            scale = float(d.mean()) if len(d) and d.mean() > 0 else np.nan
            rmse = lambda f: float(np.sqrt(np.mean((np.asarray(f) - y) ** 2)))
            mae = lambda f: float(np.mean(np.abs(np.asarray(f) - y)))
            base_rmse = rmse(sn)
            rec = {"resource": res, "region": rid,
                   "SeasonalNaive_RMSE": base_rmse, "SeasonalNaive_MAE": mae(sn),
                   "SeasonalNaive_MASE": mae(sn) / scale if scale == scale else np.nan}
            for m in ["SARIMA", "xLSTM", "TimesFM", "Nexus"]:
                f = base.get((m, res), {}).get(rid)
                if f is None:
                    continue
                rec[f"{m}_RMSE"] = rmse(f)
                rec[f"{m}_MASE"] = mae(f) / scale if scale == scale else np.nan
                rec[f"{m}_skill"] = 1 - rmse(f) / base_rmse if base_rmse > 0 else np.nan
            rows.append(rec)
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(AUDIT, "part3_seasonal_naive_skill.csv"), index=False)

    rng = np.random.default_rng(42)
    out = []
    for res, g in D.groupby("resource"):
        print(f"\n  --- {res}  (n={len(g)} regions)")
        print(f"    {'model':14s} {'RMSE':>10s} {'MASE':>7s} {'skill':>8s} {'95% CI':>20s}  verdict")
        allm = ["SeasonalNaive", "SARIMA", "xLSTM", "TimesFM", "Nexus"]
        for m in allm:
            rc, mc = f"{m}_RMSE", f"{m}_MASE"
            if rc not in g:
                continue
            sk = g[f"{m}_skill"].dropna().values if f"{m}_skill" in g else np.array([0.0] * len(g))
            if m == "SeasonalNaive":
                sk = np.zeros(len(g))
            bs = np.array([rng.choice(sk, len(sk), replace=True).mean() for _ in range(2000)]) if len(sk) else np.array([np.nan])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            v = "WORSE than copying last year" if sk.mean() < 0 else ("baseline" if m == "SeasonalNaive" else "beats seasonal-naive")
            print(f"    {m:14s} {g[rc].mean():10.3f} {g[mc].mean():7.3f} {sk.mean():+8.4f} "
                  f"[{lo:+7.4f},{hi:+7.4f}]  {v}")
            out.append({"resource": res, "model": m, "RMSE": round(float(g[rc].mean()), 4),
                        "MASE": round(float(g[mc].mean()), 4), "skill": round(float(sk.mean()), 5),
                        "skill_ci_lo": round(float(lo), 5), "skill_ci_hi": round(float(hi), 5),
                        "worse_than_seasonal_naive": bool(sk.mean() < 0)})
    R["3_skill_mase"] = out

    hdr("3.4 DOES NEXUS'S ADVANTAGE OVER SARIMA SURVIVE SEASONAL-NAIVE?")
    for res, g in D.groupby("resource"):
        if "Nexus_RMSE" not in g or "SARIMA_RMSE" not in g:
            continue
        sn, nx, sa = g.SeasonalNaive_RMSE, g.Nexus_RMSE, g.SARIMA_RMSE
        ranks = pd.DataFrame({"SeasonalNaive": sn, "Nexus": nx, "SARIMA": sa}).rank(axis=1).mean()
        best = ranks.idxmin()
        try:
            _, p = stats.wilcoxon(nx, sn)
        except Exception:
            p = np.nan
        print(f"  {res:12s} mean RMSE  seasonal-naive={sn.mean():9.3f}  SARIMA={sa.mean():9.3f}  "
              f"Nexus={nx.mean():9.3f}")
        print(f"               mean rank of 3: {dict(ranks.round(3))}  -> best={best}")
        print(f"               Nexus vs seasonal-naive Wilcoxon p={p:.4g}  "
              f"({'Nexus better' if nx.mean()<sn.mean() else 'seasonal-naive better'})")
        R.setdefault("3.4_vs_seasonal_naive", []).append(
            {"resource": res, "seasonal_naive_RMSE": round(float(sn.mean()), 4),
             "SARIMA_RMSE": round(float(sa.mean()), 4), "Nexus_RMSE": round(float(nx.mean()), 4),
             "best_of_three": best, "nexus_vs_snaive_p": float(p)})
    return D


# =============================================================== PART 4
def blend_nodes(node_base, valid, src, dst, w, a_vec):
    N, H = node_base.shape
    num, den = np.zeros((N, H)), np.zeros(N)
    good = valid[src]
    se, de, we = src[good], dst[good], w[good]
    np.add.at(num, de, we[:, None] * node_base[se])
    np.add.at(den, de, we)
    pos = den[:, None] > 0
    nbr = np.where(pos, num / np.where(pos, den[:, None], 1.0), node_base)
    a_eff = np.where(den > 0, a_vec, 0.0)
    return (1 - a_eff)[:, None] * node_base + a_eff[:, None] * nbr


def part4(base, actuals, series, sid, zone, basin):
    hdr("4.1 / 4.2 CONDITIONS INCLUDING equal_weight_neighbour (was COND_EQUAL_WEIGHT_NEIGHBOR=None)")
    Wc = torch.load(os.path.join(ROOT, "data", "weights", "gat_weights_colocation.pt"),
                    map_location="cpu", weights_only=False)
    Wr = torch.load(os.path.join(ROOT, "data", "weights", "gat_weights_random_control.pt"),
                    map_location="cpu", weights_only=False)

    def struct(W):
        s = W["site_id"].numpy().astype(int)
        att = W["site_attention_received"].numpy().astype(float)
        ei = W["attention_edge_index"].numpy(); ew = W["attention_edge_weight"].numpy().astype(float)
        keep = ei[0] != ei[1]; ei, ew = ei[:, keep], ew[keep]
        rng = att.max() - att.min()
        a = ALPHA_MAX * ((att - att.min()) / rng if rng > 0 else np.zeros_like(att))
        return dict(sid=s, a=a, src=ei[0], dst=ei[1], w=ew)

    SC, SR = struct(Wc), struct(Wr)
    # alpha-calibrated random control: scale so mean effective alpha matches colocation
    def eff_alpha(S):
        den = np.bincount(S["dst"], weights=S["w"], minlength=len(S["sid"]))
        return np.where(den > 0, S["a"], 0.0).mean()
    scale = eff_alpha(SC) / max(eff_alpha(SR), 1e-12)
    SRC_CAL = dict(SR); SRC_CAL["a"] = np.clip(SR["a"] * scale, 0, ALPHA_MAX)

    REGMAP = {"electricity": zone, "carbon": zone, "water": basin}
    rows = []
    for res in ["electricity", "carbon", "water"]:
        regs = np.array([REGMAP[res].get(int(s)) for s in sid], dtype=object)
        A = actuals.get(res, {})
        for model in ["SARIMA", "xLSTM", "TimesFM", "Nexus"]:
            bz = base.get((model, res), {})
            if not bz:
                continue
            N = len(sid)
            nb_ = np.full((N, HOLDOUT), np.nan); act = np.full((N, HOLDOUT), np.nan)
            for i, r in enumerate(regs):
                if r in bz: nb_[i] = bz[r]
                if r in A:  act[i] = A[r]
            valid = ~np.isnan(nb_).any(axis=1)
            base0 = np.nan_to_num(nb_)
            conds = {
                "unweighted": base0,
                "gat_weighted": blend_nodes(base0, valid, SC["src"], SC["dst"], SC["w"], SC["a"]),
                "random_control": blend_nodes(base0, valid, SR["src"], SR["dst"], SR["w"], SR["a"]),
                "random_control_calibrated": blend_nodes(base0, valid, SRC_CAL["src"], SRC_CAL["dst"],
                                                         SRC_CAL["w"], SRC_CAL["a"]),
                "equal_weight_neighbour": blend_nodes(base0, valid, SC["src"], SC["dst"],
                                                      np.ones_like(SC["w"]), SC["a"]),
            }
            # boundary mask for this resource
            live = regs[SC["src"]] != regs[SC["dst"]]
            bmask = np.zeros(N, bool); bmask[SC["src"][live]] = True; bmask[SC["dst"][live]] = True
            ok = valid & ~np.isnan(act).any(axis=1)
            for cond, P in conds.items():
                for stratum, m in [("all", ok), ("boundary", ok & bmask), ("non_boundary", ok & ~bmask)]:
                    if not m.any():
                        continue
                    e = P[m] - act[m]
                    scales = []
                    for r in regs[m]:
                        s_ = series[res].get(r)
                        if s_ is None or len(s_) < SEASON + HOLDOUT + 1:
                            scales.append(np.nan); continue
                        tr = s_[:-HOLDOUT]; d = np.abs(tr[SEASON:] - tr[:-SEASON])
                        d = d[np.isfinite(d)]
                        scales.append(d.mean() if len(d) and d.mean() > 0 else np.nan)
                    sc = np.asarray(scales, float)
                    mae_i = np.abs(e).mean(axis=1)
                    rmse_i = np.sqrt((e ** 2).mean(axis=1))
                    sn_rmse = []
                    for r in regs[m]:
                        s_ = series[res].get(r); tr = s_[:-HOLDOUT]
                        sn = tr[-SEASON:][:HOLDOUT]
                        sn_rmse.append(np.sqrt(np.mean((sn - A[r]) ** 2)))
                    sn_rmse = np.asarray(sn_rmse, float)
                    rows.append({"resource": res, "model": model, "condition": cond,
                                 "stratum": stratum, "n_sites": int(m.sum()),
                                 "RMSE": round(float(rmse_i.mean()), 4),
                                 "MAE": round(float(mae_i.mean()), 4),
                                 "MASE": round(float(np.nanmean(mae_i / sc)), 4),
                                 "skill_vs_snaive": round(float(np.nanmean(1 - rmse_i / sn_rmse)), 5)})
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(AUDIT, "part4_conditions_by_model.csv"), index=False)
    print(D[D.stratum == "all"].pivot_table(index=["resource", "condition"], columns="model",
                                            values="RMSE").round(2).to_string())

    hdr("5.4 STRATIFIED  (non_boundary MUST equal unweighted -- verification)")
    bad = 0
    for (res, model), g in D.groupby(["resource", "model"]):
        u = g[(g.condition == "unweighted") & (g.stratum == "non_boundary")]
        for cond in ["gat_weighted", "equal_weight_neighbour"]:
            c = g[(g.condition == cond) & (g.stratum == "non_boundary")]
            if u.empty or c.empty:
                continue
            d = abs(float(u.RMSE.iloc[0]) - float(c.RMSE.iloc[0]))
            if d > 1e-6:
                bad += 1
                print(f"  *** LOUD FLAG: {res}/{model}/{cond} non-boundary RMSE differs from "
                      f"unweighted by {d:.6f} -- indicates a BUG, not a finding.")
    if not bad:
        print("  [OK] For every resource x model, non-boundary strata are numerically identical to")
        print("       unweighted under gat_weighted and equal_weight_neighbour, as required.")
    R["5.4_nonboundary_identical"] = (bad == 0)
    R["4_conditions"] = D[D.stratum == "all"].to_dict("records")
    return D


# =============================================================== main
def main():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    ns = {"__file__": os.path.join(ROOT, "scripts", "run_comparison.py")}
    exec(open(ns["__file__"], encoding="utf-8").read().split("def main()")[0], ns)
    al = pd.read_parquet(os.path.join(ROOT, "data", "aligned_dataset.parquet"))
    nodes = al[al.is_node]
    zone = dict(zip(nodes.site_id.astype(int), nodes.grid_zone_id))
    basin, _ = ns["build_basin_map"](nodes)

    rs = json.load(open(os.path.join(PARTB, "region_series_cache.json")))
    series = {r: {k: np.asarray(v["values"] if isinstance(v, dict) and "values" in v else v, float)
                  for k, v in d.items()} for r, d in rs.items()}
    actuals = {r: {k: v[-HOLDOUT:] for k, v in d.items() if len(v) >= HOLDOUT}
               for r, d in series.items()}

    pf = pd.read_csv(os.path.join(PARTB, "per_region_base_forecasts.csv"), parse_dates=["timestamp"])
    base = {}
    for (m, res, rid), g in pf.groupby(["model", "resource", "region_id"]):
        v = g.sort_values("timestamp").value.values[:HOLDOUT]
        if len(v) == HOLDOUT:
            base.setdefault((m, res), {})[rid] = v

    W = torch.load(os.path.join(ROOT, "data", "weights", "gat_weights_colocation.pt"),
                   map_location="cpu", weights_only=False)
    sid = W["site_id"].numpy().astype(int)

    part1(series, nodes, zone, basin)
    part2(base, series, actuals)
    part3(base, series, actuals)
    part4(base, actuals, series, sid, zone, basin)

    # ---------------------------------------------------- 6.1 / 6.2
    p5 = os.path.join(AUDIT, "part5_results.json")
    if os.path.exists(p5):
        R["part5"] = json.load(open(p5))
    json.dump(R, open(os.path.join(AUDIT, "results.json"), "w"), indent=2, default=str)

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()[:16]
    inputs = {}
    for rel in ["data/aligned_dataset.parquet", "data/weights/gat_weights_colocation.pt",
                "data/weights/gat_weights_random_control.pt", "data/g3p/G3P_v1.12_tws_rivbas.csv"]:
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            inputs[rel] = {"sha256_16": sha(p), "bytes": os.path.getsize(p)}
    man = {"seeds": {"SEED": 42},
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "torch": torch.__version__,
                        "scipy": __import__("scipy").__version__},
           "inputs_sha256": inputs,
           "models_ran": sorted({k[0] for k in base}),
           "nexus_provenance": R.get("2_nexus_provenance"),
           "holdout_months": HOLDOUT, "min_zone_months": 36,
           "runtime_seconds": round(time.time() - T0, 1)}
    json.dump(man, open(os.path.join(AUDIT, "run_manifest.json"), "w"), indent=2, default=str)
    print(f"\nwrote {AUDIT}/results.json and run_manifest.json")


if __name__ == "__main__":
    main()
