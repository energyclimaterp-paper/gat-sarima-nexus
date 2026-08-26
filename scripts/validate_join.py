"""Replicate notebook §3b/§3c exactly and report how many series survive."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

MIN_ZONE_MONTHS = 36
def _norm(s):
    return (str(s).strip().lower().replace("&","and").replace(".","").replace(",","").replace("  "," "))

def load_ember_region(path, region):
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in ["State","State code","Area","ISO 3 code","Area type",
                           "Date","Category","Variable","Unit","Value"] if c in cols]
    df = pd.read_csv(path, usecols=usecols); df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    gen = df[(df["Category"]=="Electricity generation") & (df["Variable"].str.lower()=="total generation")
             & (df["Unit"].isin(["GWh","TWh"]))].copy()
    gen["electricity_gwh"] = np.where(gen["Unit"]=="TWh", gen["Value"]*1000.0, gen["Value"])
    car = df[(df["Category"]=="Power sector emissions") & (df["Variable"]=="CO2 intensity")].copy()
    car["co2_intensity"] = car["Value"]
    if region in ("US","India"):
        iso = "USA" if region=="US" else "IND"
        for g in (gen, car):
            g["zone_id"] = iso + "|" + g["State"].map(_norm)
            g.drop(g.index[g["State"].astype(str).str.lower().str.contains("total")], inplace=True)
    else:
        gen = gen[gen["Area type"]=="Country or economy"]; car = car[car["Area type"]=="Country or economy"]
        gen["zone_id"] = gen["ISO 3 code"].astype(str); car["zone_id"] = car["ISO 3 code"].astype(str)
    gen = gen.dropna(subset=["Value","zone_id","Date"]); car = car.dropna(subset=["Value","zone_id","Date"])
    g = gen.groupby(["zone_id","Date"], as_index=False)["electricity_gwh"].mean()
    c = car.groupby(["zone_id","Date"], as_index=False)["co2_intensity"].mean()
    print(f"  {region:6s}: {len(g):6d} gen rows, {len(c):6d} carbon rows, {g.zone_id.nunique():4d} zones")
    return pd.merge(g, c, on=["zone_id","Date"], how="outer")

E = "data/ember"
print("Loading Ember (this reads ~160 MB)...")
ember = pd.concat([
    load_ember_region(f"{E}/us_monthly_full_release_long_format.csv","US"),
    load_ember_region(f"{E}/europe_monthly_full_release_long_format.csv","EU"),
    load_ember_region(f"{E}/india_monthly_full_release_long_format.csv","India"),
], ignore_index=True)
print(f"\nember panel: {len(ember):,} rows | {ember.zone_id.nunique()} zones | "
      f"{ember.Date.min().date()} .. {ember.Date.max().date()}")

aligned = pd.read_parquet("data/aligned_dataset.parquet")
zones_with_nodes = {z for z in pd.unique(aligned.loc[aligned["is_node"],"grid_zone_id"]) if isinstance(z,str)}
print(f"zones containing >=1 graph node: {len(zones_with_nodes)}")

ez = set(ember.zone_id.unique())
hit = zones_with_nodes & ez
print(f"\nJOIN: {len(hit)}/{len(zones_with_nodes)} node-zones found in Ember")
miss = sorted(zones_with_nodes - ez)
if miss: print(f"  missing ({len(miss)}): {miss[:20]}{' ...' if len(miss)>20 else ''}")

region_series = {"electricity":{}, "carbon":{}, "water":{}}
for z in sorted(zones_with_nodes):
    sub = ember[ember.zone_id==z].set_index("Date").sort_index()
    if sub.empty: continue
    for rk, col in [("electricity","electricity_gwh"),("carbon","co2_intensity")]:
        s = sub[col].dropna(); s = s[~s.index.duplicated(keep="first")]
        if len(s) >= MIN_ZONE_MONTHS:
            region_series[rk][z] = s
print(f"\nusable electricity series (>= {MIN_ZONE_MONTHS} months): {len(region_series['electricity'])}")
print(f"usable carbon series                          : {len(region_series['carbon'])}")
for rk in ("electricity","carbon"):
    if region_series[rk]:
        L = [len(s) for s in region_series[rk].values()]
        print(f"  {rk:12s} months per zone: min={min(L)} median={int(np.median(L))} max={max(L)}")

# ---- water
tws = pd.read_csv("data/g3p/G3P_v1.12_tws_rivbas.csv", parse_dates=["time [yyyy-mm-dd]"])
tws = tws.rename(columns={"time [yyyy-mm-dd]":"date"}).set_index("date").sort_index()
basins = [c.replace(" [mm]","") for c in tws.columns
          if c.endswith("[mm]") and not c.startswith("uncertainty") and c != "global [mm]"]
print(f"\nG3P: {len(basins)} basin columns, {len(tws)} monthly rows "
      f"({tws.index.min().date()} .. {tws.index.max().date()})")
print(f"  sample basins: {basins[:6]}")
