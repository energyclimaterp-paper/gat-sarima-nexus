# Exact inputs and outputs

Traced from the notebook source, not inferred.

## Inputs

§1 resolves 8 paths but enforces only 7 (`REQUIRED`). You have 3 of them.

| key | file pattern | required | status |
|---|---|---|---|
| `aligned` | `aligned_dataset.parquet` | yes | **have** |
| `w_coloc` | `gat_weights_colocation.pt` | yes | **have** |
| `w_random` | `gat_weights_random_control.pt` | yes | **have** |
| `ember_us` | `*us_monthly*.csv` | yes | missing |
| `ember_eu` | `*europe_monthly*.csv` | yes | missing |
| `ember_in` | `*india_monthly*.csv` | yes | missing |
| `g3p_tws` | `*tws_rivbas*.csv` | yes | missing |
| `nodes` | `graph_nodes.parquet` | **no** | not needed — resolved but never read |

### What each input is actually used for

**`aligned_dataset.parquet`** (§1 hard-checks `site_id`, `grid_zone_id`, `pfaf_id`)
- `site_id` + `grid_zone_id` -> `zone_of_site` (join key for electricity/carbon)
- `latitude`/`longitude` -> nearest-basin assignment (`basin_of_site`)
- `is_node` -> restricts to the 4,887 graph nodes
- `pfaf_id` -> presence-checked only, value never read

**`gat_weights_*.pt`** — only 3 of the 11 keys are consumed by §4:
`site_id`, `site_attention_received`, `attention_edge_index`, `attention_edge_weight`.
`embeddings` is **never used** by this notebook. Neither is `emb_dim`, `epochs`,
`final_reconstruction_mse`, or `note`.

**Ember CSVs** — §12 needs these columns:
`State`, `State code`, `Area`, `ISO 3 code`, `Area type`, `Date`, `Category`,
`Variable`, `Unit`, `Value`

and filters two slices out of them:

| series | filter |
|---|---|
| electricity | `Category == "Electricity generation"` AND `Variable.lower() == "total generation"` AND `Unit in {GWh, TWh}` (TWh multiplied by 1000) |
| carbon | `Category == "Power sector emissions"` AND `Variable == "CO2 intensity"` |

then builds the zone key:

| region | zone_id |
|---|---|
| US | `"USA\|" + normalise(State)` — rows whose State contains "total" dropped |
| India | `"IND\|" + normalise(State)` |
| Europe | rows where `Area type == "Country or economy"`, `zone_id = ISO 3 code` |

`normalise` = lowercase, strip, `&`->`and`, drop `.` and `,`.

**`tws_rivbas.csv` (G3P)** — §11 needs:
- a `time [yyyy-mm-dd]` column (parsed to the index)
- basin columns named `<Basin> [mm]`
- columns starting `uncertainty` and the `global [mm]` column are excluded
- basin names must match the ~100 hardcoded keys in `BASIN_CENTROIDS` (§0)

## The join key, verified against your data

`region_series` is keyed by `zone_id`, which must equal `grid_zone_id` in the
parquet. Among the 4,887 graph nodes:

| format | distinct | sites | examples |
|---|---|---|---|
| `ISO3` (Europe) | 34 | 1,820 | `AUT`, `BEL`, `DEU`, `ESP` |
| `USA\|state` / `IND\|state` | 60 | 3,067 | `USA\|virginia`, `IND\|delhi` |

**94 zones need a monthly series.** Any Ember file can be validated against this
list before running anything — if the derived `zone_id` values don't intersect
these 94, the join silently produces zero series.

Gate: a zone needs `MIN_ZONE_MONTHS = 36` months to qualify, and the last
`HOLDOUT = 12` become ground truth.

## Outputs

All to `WORK` (`outputs/` locally, `/kaggle/working` on Kaggle).

**§9 — forecasts**
| file | contents |
|---|---|
| `forecasts_unweighted.csv` | per-site base forecasts |
| `forecasts_gat_weighted.csv` | per-site, blended over real co-location edges |
| `forecasts_random_control_weighted.csv` | per-site, blended over random edges |
| `per_region_base_forecasts.csv` | `region_id, resource, model, timestamp, value` |
| `forecast_failures_log.csv` | `region_id, resource, model, reason` |
| `weight_injection_log.json` | run config + changed-fraction per condition |

**§10 — scoring**
| file | contents |
|---|---|
| `step5_metrics_site_resource.csv` | MAE/RMSE/MAPE per site x resource x condition |
| `step5_aggregate_by_resource.csv` | means per resource x condition |
| `step5_significance.csv` | paired t-test + Wilcoxon, gat vs unweighted and vs random |
| `step5_diagram_versions.csv` | oracle best-single / simple-average / graph-weighted / random |

## The gap, stated plainly

Everything the graph side needs, you have. What is missing is the **time axis**:
94 zone-level monthly series (electricity + CO2 intensity) and the basin-level
water series. Neither `aligned_dataset.parquet` nor the `.pt` files contain one —
the parquet holds a single snapshot value per site, and every tensor in the `.pt`
files is indexed by node (4887) or edge (71161), never by month.
