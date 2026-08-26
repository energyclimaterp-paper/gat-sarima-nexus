# Data dictionary

## `data/aligned_dataset.parquet`

6,131 rows x 16 columns. One row = one data-centre site.

| column | dtype | nulls | meaning |
|---|---|---|---|
| `site_id` | int64 | 0 | primary key; joins to `site_id` in the `.pt` payloads |
| `latitude` | float64 | 0 | WGS84 |
| `longitude` | float64 | 0 | WGS84 |
| `country_iso3` | str | 16 | ISO 3166-1 alpha-3 |
| `country` | str | 16 | country name |
| `name_1` | str | 16 | admin level 1 (state / province / region) |
| `pfaf_id` | float64 | 1 | Pfafstetter basin code (HydroBASINS) |
| `water_stress_bws` | float64 | 1 | WRI Aqueduct baseline water stress. **Static** — climatology, no time axis, so not forecastable |
| `grid_zone_id` | str | 1238 | EMBER zone key, e.g. `ESP`, `USA\|california` |
| `electricity_gwh` | float64 | 1244 | zone monthly total generation, GWh (TWh rows converted x1000) |
| `co2_intensity` | float64 | 1244 | zone power-sector CO2 intensity, gCO2/kWh |
| `name` | str | 0 | facility name |
| `company` | str | 21 | operator |
| `city` | str | 0 | city |
| `temperature_c` | float64 | 1244 | Open-Meteo reading. **Snapshot** — one value, no monthly history, so not forecastable |
| `is_node` | bool | 0 | 4,887 True — exactly the sites present in the GAT payloads |

Coverage: 96 distinct grid zones, 141 countries. The 1,244-null block is one
consistent group — sites with no EMBER zone match lose grid *and* temperature together.

## `data/weights/*.pt`

`torch.load(path, map_location="cpu", weights_only=False)` returns a plain dict:

| key | type | shape / value |
|---|---|---|
| `site_id` | int64 tensor | (4887,) |
| `embeddings` | float32 tensor | (4887, 8) |
| `site_attention_received` | float32 tensor | (4887,) — scaled to the blend weight `a` |
| `attention_edge_index` | int64 tensor | (2, 71161) — `[src; dst]` |
| `attention_edge_weight` | float32 tensor | (71161,) |
| `node_features_used` | list | `['water_stress_bws','temperature_c','electricity_gwh','co2_intensity']` |
| `final_reconstruction_mse` | float | **0.0333** coloc / **0.4741** random |
| `emb_dim` / `epochs` / `seed` | int | 8 / 400 / 42 |
| `note` | str | "Embeddings are a spatial-co-location-prior representation, NOT a physical model." |

Both files are byte-for-byte the same *shape* — same node count, same edge count. Only
the edge wiring (and therefore what was learned) differs. That is what makes the
random-control comparison a fair one.

Of the 71,161 edges, 4,887 are self-loops (one per node); the notebook strips these in
§4, leaving 66,274 real edges — average degree ≈ 13.6.

## Inputs NOT in this repo

The notebook resolves these by filename search under `/kaggle/input`. Supply them and
point §1 at their location:

| expected filename pattern | what it is |
|---|---|
| `graph_nodes.parquet` | node table used when building the graph |
| `*us_monthly*.csv` | EMBER US monthly electricity + emissions |
| `*europe_monthly*.csv` | EMBER Europe monthly |
| `*india_monthly*.csv` | EMBER India monthly |
| `*tws_rivbas*.csv` | G3P total water storage anomaly per river basin, monthly |

Without the EMBER and G3P files, §3 produces empty `region_series` and every downstream
section has nothing to forecast. The `.pt` files and `aligned_dataset.parquet` alone are
enough to inspect the graph, but not to reproduce the forecasts.

## Key constants (notebook §0)

```
SEED = 42            HOLDOUT = 12 months      LOOKBACK = 12
MIN_ZONE_MONTHS = 36 ALPHA_MAX = 0.5          BASIN_MAX_KM = 2500
MODELS = ["SARIMA", "Nexus"]
RESOURCES = {"electricity": "zone", "carbon": "zone", "water": "basin"}
```
