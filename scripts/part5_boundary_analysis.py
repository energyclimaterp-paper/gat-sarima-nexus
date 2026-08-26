"""PART 5 -- Is co-location viable at all? Boundary-case analysis.

No forecaster reruns. Everything here is graph geometry + the training portion
of the region series.

Key premise being tested: forecasts are produced per REGION and copied to every
site in that region, so an edge whose endpoints share a forecasting region
averages a series with itself and cannot change any number. The graph can only
act through cross-region ("live") edges.
"""
from __future__ import annotations
import json, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTA = os.path.join(ROOT, "outputs_v2", "partA")
PARTB = os.path.join(ROOT, "outputs_v2", "partB")
AUDIT = os.path.join(ROOT, "outputs_v2", "audit")
os.makedirs(AUDIT, exist_ok=True)
HOLDOUT = 12
RESULTS = {}


def hdr(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ------------------------------------------------------------------ load state
def load():
    al = pd.read_parquet(os.path.join(ROOT, "data", "aligned_dataset.parquet"))
    nodes = al[al.is_node].reset_index(drop=True)
    W = torch.load(os.path.join(ROOT, "data", "weights", "gat_weights_colocation.pt"),
                   map_location="cpu", weights_only=False)
    sid = W["site_id"].numpy().astype(int)
    ei = W["attention_edge_index"].numpy()
    ei = ei[:, ei[0] != ei[1]]                      # drop self-loops

    # region maps, exactly as Part B builds them
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    ns = {"__file__": os.path.join(ROOT, "scripts", "run_comparison.py")}
    exec(open(ns["__file__"], encoding="utf-8").read().split("def main()")[0], ns)
    zone = dict(zip(nodes.site_id.astype(int), nodes.grid_zone_id))
    basin, tws = ns["build_basin_map"](nodes)

    rs = json.load(open(os.path.join(PARTB, "region_series_cache.json")))
    series = {r: {k: np.asarray(v["values"] if isinstance(v, dict) and "values" in v else v, float)
                  for k, v in d.items()} for r, d in rs.items()}

    meta = nodes.set_index("site_id")
    lat = meta.loc[sid, "latitude"].values
    lon = meta.loc[sid, "longitude"].values
    pfaf = meta.loc[sid, "pfaf_id"].values
    zarr = np.array([zone.get(int(s)) for s in sid], dtype=object)
    barr = np.array([basin.get(int(s)) for s in sid], dtype=object)
    ctry = meta.loc[sid, "country"].values
    return dict(sid=sid, ei=ei, lat=lat, lon=lon, pfaf=pfaf, zone=zarr, basin=barr,
                country=ctry, series=series, tws=tws, nodes=nodes)


REGION_ATTR = {"electricity": "zone", "carbon": "zone", "water": "basin"}


# --------------------------------------------------------------- 5.1 / 5.2 / 5.3
def taxonomy(S):
    hdr("5.1 EDGE TAXONOMY  (INERT = same forecasting region; cannot affect any result)")
    ei = S["ei"]
    same_zone_edge = S["zone"][ei[0]] == S["zone"][ei[1]]
    same_pfaf_edge = S["pfaf"][ei[0]] == S["pfaf"][ei[1]]
    rows = []
    for res in ["electricity", "carbon", "water"]:
        reg = S[REGION_ATTR[res]]
        live = reg[ei[0]] != reg[ei[1]]
        n = ei.shape[1]
        for origin, mask in [("shared-zone", same_zone_edge),
                             ("shared-basin(pfaf)", same_pfaf_edge),
                             ("ALL", np.ones(n, bool))]:
            k = int(mask.sum())
            lv = int((live & mask).sum())
            rows.append({"resource": res, "edge_origin": origin, "edges": k,
                         "LIVE": lv, "INERT": k - lv,
                         "live_%": round(100 * lv / k, 3) if k else np.nan})
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    print("\nINERT edges average a region series with itself: they cannot change any number.")
    T.to_csv(os.path.join(AUDIT, "part5_1_edge_taxonomy.csv"), index=False)
    RESULTS["5.1_edge_taxonomy"] = T.to_dict("records")

    hdr("5.2 NODE TAXONOMY  (BOUNDARY NODE = has >=1 LIVE edge)")
    rows, bmasks = [], {}
    N = len(S["sid"])
    for res in ["electricity", "carbon", "water"]:
        reg = S[REGION_ATTR[res]]
        live = reg[ei[0]] != reg[ei[1]]
        b = np.zeros(N, bool)
        b[ei[0][live]] = True
        b[ei[1][live]] = True
        bmasks[res] = b
        rows.append({"resource": res, "nodes": N, "boundary_nodes": int(b.sum()),
                     "boundary_%": round(100 * b.mean(), 2),
                     "n_countries_with_boundary": int(pd.Series(S["country"][b]).nunique()),
                     "n_regions_with_boundary": int(pd.Series(reg[b]).nunique())})
    T2 = pd.DataFrame(rows)
    print(T2.to_string(index=False))
    for res in ["electricity", "water"]:
        top = pd.Series(S["country"][bmasks[res]]).value_counts().head(6)
        print(f"\n  {res}: boundary nodes by country (top 6)\n" +
              "\n".join(f"    {k:24s} {v}" for k, v in top.items()))
    T2.to_csv(os.path.join(AUDIT, "part5_2_node_taxonomy.csv"), index=False)
    RESULTS["5.2_node_taxonomy"] = T2.to_dict("records")

    hdr("5.3 REGION-PAIR TAXONOMY  (the true information unit of the graph)")
    rows, pairs_by_res = [], {}
    for res in ["electricity", "carbon", "water"]:
        reg = S[REGION_ATTR[res]]
        live = reg[ei[0]] != reg[ei[1]]
        a, b = reg[ei[0][live]], reg[ei[1][live]]
        ordered = set(zip(a, b))
        unordered = set(tuple(sorted((x, y), key=str)) for x, y in zip(a, b))
        cnt = pd.Series([tuple(sorted((x, y), key=str)) for x, y in zip(a, b)]).value_counts()
        pairs_by_res[res] = cnt
        n_reg = len(S["series"].get(res, {}))
        rows.append({"resource": res, "regions": n_reg,
                     "ordered_pairs": len(ordered), "unordered_pairs": len(unordered),
                     "max_possible_unordered": n_reg * (n_reg - 1) // 2,
                     "pct_of_possible": round(100 * len(unordered) / max(n_reg * (n_reg - 1) // 2, 1), 3)})
    T3 = pd.DataFrame(rows)
    print(T3.to_string(index=False))
    for res in ["electricity", "water"]:
        print(f"\n  {res}: top bridged pairs by live-edge count")
        for (x, y), c in pairs_by_res[res].head(20 if res == "electricity" else 10).items():
            print(f"    {str(x):34s} <-> {str(y):34s} {c}")
    T3.to_csv(os.path.join(AUDIT, "part5_3_region_pairs.csv"), index=False)
    RESULTS["5.3_region_pairs"] = T3.to_dict("records")
    return bmasks, pairs_by_res


# ----------------------------------------------------------- 5.5 / 5.6 viability
def viability(S, pairs_by_res):
    hdr("5.5 CO-LOCATION VIABILITY  (independent of the GAT; TRAINING portion only)")
    print("Do live-edge-connected region pairs co-move more than unconnected pairs?")
    out = []
    detr_cache, cent = {}, {}
    for res in ["electricity", "carbon", "water"]:
        ser = S["series"].get(res, {})
        reg_ids = sorted(ser)
        if len(reg_ids) < 4:
            continue
        # detrended training portion (first difference removes level+trend)
        D = {}
        for r in reg_ids:
            v = ser[r][:-HOLDOUT]
            v = v[np.isfinite(v)]
            D[r] = np.diff(v) if len(v) > 2 else None
        detr_cache[res] = D
        # region centroids from member sites
        attr = REGION_ATTR[res]
        cen = {}
        for r in reg_ids:
            m = S[attr] == r
            if m.sum():
                cen[r] = (float(np.mean(S["lat"][m])), float(np.mean(S["lon"][m])))
        cent[res] = cen

        connected = set(pairs_by_res[res].index)
        rows = []
        for i in range(len(reg_ids)):
            for j in range(i + 1, len(reg_ids)):
                a, b = reg_ids[i], reg_ids[j]
                da, db = D.get(a), D.get(b)
                if da is None or db is None:
                    continue
                n = min(len(da), len(db))
                if n < 24:
                    continue
                rho, _ = stats.spearmanr(da[-n:], db[-n:])
                if not np.isfinite(rho):
                    continue
                key = tuple(sorted((a, b), key=str))
                d_km = (haversine(*cen[a], *cen[b]) if a in cen and b in cen else np.nan)
                rows.append({"a": a, "b": b, "rho": rho,
                             "connected": key in connected, "dist_km": d_km})
        P = pd.DataFrame(rows)
        if P.empty:
            continue
        c, u = P[P.connected].rho, P[~P.connected].rho
        if len(c) < 3 or len(u) < 3:
            print(f"  {res}: too few pairs to test (connected={len(c)}, unconnected={len(u)})")
            continue
        U, p = stats.mannwhitneyu(c, u, alternative="two-sided")
        # rank-biserial effect size
        rb = 2 * U / (len(c) * len(u)) - 1
        verdict = ("connected pairs co-move MORE" if c.mean() > u.mean()
                   else "connected pairs co-move LESS")
        sig = "SIGNIFICANT" if p < 0.05 else "NOT significant"
        print(f"\n  {res}: connected n={len(c)} mean rho={c.mean():+.4f} | "
              f"unconnected n={len(u)} mean rho={u.mean():+.4f}")
        print(f"     Mann-Whitney U p={p:.4g}  rank-biserial={rb:+.4f}  -> {verdict}, {sig}")
        out.append({"resource": res, "n_connected": len(c), "n_unconnected": len(u),
                    "mean_rho_connected": round(float(c.mean()), 5),
                    "mean_rho_unconnected": round(float(u.mean()), 5),
                    "mannwhitney_p": float(p), "rank_biserial": float(rb),
                    "significant": bool(p < 0.05), "direction": verdict})
        P.to_csv(os.path.join(AUDIT, f"part5_5_pairs_{res}.csv"), index=False)

        # ---- 5.6 distance control: match each connected pair to the nearest-distance
        # unconnected pair, sampling without replacement.
        conn = P[P.connected].dropna(subset=["dist_km"]).copy()
        unc = P[~P.connected].dropna(subset=["dist_km"]).copy()
        if len(conn) >= 3 and len(unc) >= len(conn):
            unc_sorted = unc.sort_values("dist_km").reset_index(drop=True)
            avail = np.ones(len(unc_sorted), bool)
            matched = []
            for d in conn.dist_km.values:
                idx = np.where(avail)[0]
                if not len(idx):
                    break
                j = idx[np.argmin(np.abs(unc_sorted.dist_km.values[idx] - d))]
                avail[j] = False
                matched.append(unc_sorted.rho.values[j])
            matched = np.asarray(matched)
            if len(matched) >= 3:
                U2, p2 = stats.mannwhitneyu(conn.rho.values[:len(matched)], matched,
                                            alternative="two-sided")
                rb2 = 2 * U2 / (len(matched) ** 2) - 1
                surv = "SURVIVES" if p2 < 0.05 else "DOES NOT survive"
                print(f"     [5.6 distance-matched] connected mean={conn.rho.mean():+.4f} vs "
                      f"matched-unconnected mean={matched.mean():+.4f}  p={p2:.4g}  -> {surv}")
                out[-1].update({"dist_matched_p": float(p2),
                                "dist_matched_rho_connected": round(float(conn.rho.mean()), 5),
                                "dist_matched_rho_unconnected": round(float(matched.mean()), 5),
                                "survives_distance_control": bool(p2 < 0.05)})
    RESULTS["5.5_5.6_viability"] = out
    pd.DataFrame(out).to_csv(os.path.join(AUDIT, "part5_5_viability.csv"), index=False)
    return cent


# ------------------------------------------------------- 5.7 boundary geography
def geography(S, bmasks):
    hdr("5.7 BOUNDARY GEOGRAPHY  (distance to nearest cross-region neighbour)")
    ei = S["ei"]
    rows, allrec = [], []
    for res in ["electricity", "carbon", "water"]:
        reg = S[REGION_ATTR[res]]
        live = reg[ei[0]] != reg[ei[1]]
        src, dst = ei[0][live], ei[1][live]
        d = haversine(S["lat"][src], S["lon"][src], S["lat"][dst], S["lon"][dst])
        if not len(d):
            continue
        best = {}
        for s_, t_, dd in zip(src, dst, d):
            for a, b in ((s_, t_), (t_, s_)):
                if a not in best or dd < best[a][0]:
                    best[a] = (dd, reg[a], reg[b])
        nd = np.array([v[0] for v in best.values()])
        over = int((d > 500).sum())
        rows.append({"resource": res, "live_edges": int(live.sum()),
                     "boundary_nodes": len(best),
                     "median_km_to_nearest_cross_region": round(float(np.median(nd)), 1),
                     "p90_km": round(float(np.percentile(nd, 90)), 1),
                     "max_km": round(float(nd.max()), 1),
                     "live_edges_over_500km": over,
                     "pct_live_over_500km": round(100 * over / len(d), 2)})
        for a, (dd, ra, rb) in best.items():
            allrec.append({"resource": res, "site_id": int(S["sid"][a]),
                           "km_to_nearest_cross_region": round(float(dd), 2),
                           "region": ra, "bridged_to": rb})
    T = pd.DataFrame(rows)
    print(T.to_string(index=False))
    for r in rows:
        if r["live_edges_over_500km"]:
            print(f"  ** WARNING: {r['resource']}: {r['live_edges_over_500km']} live edges span >500 km "
                  f"({r['pct_live_over_500km']}%) -- unlikely to represent shared resource context.")
    pd.DataFrame(allrec).to_csv(os.path.join(AUDIT, "part5_7_boundary_geography.csv"), index=False)
    T.to_csv(os.path.join(AUDIT, "part5_7_summary.csv"), index=False)
    RESULTS["5.7_boundary_geography"] = T.to_dict("records")


# ------------------------------------------------- 5.8 basin definition agreement
def basin_consistency(S):
    hdr("5.8 BASIN DEFINITION CONSISTENCY  (Aqueduct pfaf_id vs G3P nearest-centroid)")
    nb = json.load(open(os.path.join(ROOT, "notebooks", "gat_weighted_forecasting.ipynb"),
                       encoding="utf-8"))
    ns = {"np": np, "pd": pd}
    exec("".join(nb["cells"][3]["source"]), ns)
    BC, MAXKM, hav = ns["BASIN_CENTROIDS"], ns["BASIN_MAX_KM"], ns["_haversine_km"]
    tws = S["tws"]
    basins = [c.replace(" [mm]", "") for c in tws.columns
              if c.endswith("[mm]") and not c.startswith("uncertainty") and c != "global [mm]"]
    cand = [b for b in basins if b in BC]
    clat = np.array([BC[b][0] for b in cand]); clon = np.array([BC[b][1] for b in cand])

    dists, assigned = [], []
    for la, lo in zip(S["lat"], S["lon"]):
        d = hav(float(la), float(lo), clat, clon)
        j = int(np.argmin(d))
        dists.append(float(d[j])); assigned.append(cand[j] if d[j] <= MAXKM else "Unknown")
    dists = np.asarray(dists); assigned = np.asarray(assigned, dtype=object)

    print(f"nearest-centroid assignment distance (km): median={np.median(dists):.0f} "
          f"p90={np.percentile(dists,90):.0f} max={dists.max():.0f}")
    far = int((dists > 500).sum())
    print(f"sites assigned to a G3P basin >500 km away: {far} / {len(dists)} "
          f"({100*far/len(dists):.1f}%)")
    if far:
        idx = np.argsort(-dists)[:10]
        print("  farthest 10:")
        for i in idx:
            print(f"    site {int(S['sid'][i]):5d}  {dists[i]:7.0f} km -> {assigned[i]}")

    # agreement: do two sites sharing a pfaf_id also share a G3P basin?
    ei = S["ei"]
    same_pfaf = S["pfaf"][ei[0]] == S["pfaf"][ei[1]]
    same_g3p = assigned[ei[0]] == assigned[ei[1]]
    agree = int((same_pfaf == same_g3p).sum())
    print(f"\nedge-level partition agreement: {agree}/{ei.shape[1]} "
          f"({100*agree/ei.shape[1]:.2f}%) of edges are classified the same way by both partitions")
    live_water = int((~same_g3p).sum())
    both = int(((~same_g3p) & (~same_pfaf)).sum())
    print(f"live water edges (G3P partition): {live_water}")
    print(f"  of which ALSO cross-basin under Aqueduct pfaf_id: {both} "
          f"({100*both/max(live_water,1):.1f}%) -- these are the ones that survive "
          f"restricting to sites where the two partitions agree")
    rec = {"median_assign_km": float(np.median(dists)), "p90_assign_km": float(np.percentile(dists, 90)),
           "max_assign_km": float(dists.max()), "sites_over_500km": far,
           "edge_partition_agreement_pct": round(100 * agree / ei.shape[1], 3),
           "live_water_edges_g3p": live_water, "live_and_cross_pfaf": both}
    RESULTS["5.8_basin_consistency"] = rec
    pd.DataFrame([rec]).to_csv(os.path.join(AUDIT, "part5_8_basin_consistency.csv"), index=False)
    far_rows = pd.DataFrame({"site_id": S["sid"], "assign_km": dists, "g3p_basin": assigned})
    far_rows[far_rows.assign_km > 500].to_csv(
        os.path.join(AUDIT, "part5_8_sites_over_500km.csv"), index=False)


if __name__ == "__main__":
    S = load()
    print(f"loaded: {len(S['sid'])} nodes, {S['ei'].shape[1]} non-self-loop edges")
    bmasks, pairs = taxonomy(S)
    viability(S, pairs)
    geography(S, bmasks)
    basin_consistency(S)
    json.dump(RESULTS, open(os.path.join(AUDIT, "part5_results.json"), "w"), indent=2, default=str)
    print(f"\nwrote {os.path.join(AUDIT,'part5_results.json')} and per-section CSVs")
