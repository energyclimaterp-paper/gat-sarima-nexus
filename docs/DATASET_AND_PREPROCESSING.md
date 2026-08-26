# Dataset and Preprocessing

## 2.1 Sources

The study joins five independent public sources around a single spatial unit — an
individual data-centre facility.

| # | Source | Release / version | Native unit | Temporal coverage | Role |
|---|---|---|---|---|---|
| 1 | Data-centre atlas | — | facility | static | site registry (name, operator, city, country, coordinates) |
| 2 | WRI Aqueduct 4.0 | `Y2023M07D05`, `baseline_annual` | hydrological basin polygon | static climatology | basin id (`pfaf_id`), admin-1 name, baseline water stress (`bws_score`) |
| 3 | Ember monthly electricity | full release, long format (US / Europe / India) | grid zone | 2001-01 → 2026-04, monthly | total generation (GWh), power-sector CO₂ intensity (gCO₂/kWh) |
| 4 | Open-Meteo | live API | facility | instantaneous snapshot | near-surface air temperature |
| 5 | G3P | v1.12, `tws_rivbas` | river basin | 2002-04 → 2023-09, monthly | total water storage anomaly (mm) |

Sources 3 and 5 are the only ones with a time axis, and therefore the only
forecastable quantities. Aqueduct water stress is a static climatology and the
Open-Meteo reading is a single snapshot; both enter the model as node features
but are explicitly excluded from forecasting rather than interpolated into
synthetic series.

## 2.2 Spatial alignment

Of **18,110** atlas records, **6,131 (33.9%)** carry usable coordinates and are
retained; the remainder cannot be spatially matched by any method and are
dropped rather than imputed. Sites are indexed `site_id` 0…6130.

Each site is then resolved against three region systems:

1. **Water basin (Aqueduct).** Point-in-polygon join of the site to the Aqueduct
   4.0 basin layer, yielding `pfaf_id`, country code, admin-1 name and
   `bws_score`. Aqueduct's `9999` no-data sentinel is converted to null.
2. **Grid zone (Ember).** The admin-1 name from step 1 is normalised
   (case-folded, punctuation stripped, `&`→`and`), passed through a manual alias
   table for known exonyms (`orissa`→`odisha`, `nct of delhi`→`delhi`,
   `district of columbia`→`washington dc`, …), then fuzzy-matched with a 0.82
   similarity cutoff. Zone keys are `USA|<state>` and `IND|<state>` for the two
   state-resolved markets and the bare ISO-3 code for European countries. Among
   graph nodes this yields 3,067 state-level and 1,822 country-level sites
   across **95 distinct zones**.
3. **River basin (G3P).** Because G3P is published per named basin rather than
   as polygons, each site is assigned to the nearest of 100 basin centroids by
   haversine distance, subject to a 2,500 km cutoff; beyond that the site is
   marked `Unknown` rather than force-matched. 24 basins contain at least one
   site.

## 2.3 Temporal alignment

All three Ember releases are already monthly, so monthly is the coarsest common
granularity and no downsampling is required. Two filters extract the target
series:

- **Electricity** — `Category = "Electricity generation"`,
  `Variable = "total generation"`, `Unit ∈ {GWh, TWh}`, with TWh rescaled ×1000.
- **Carbon** — `Category = "Power sector emissions"`, `Variable = "CO2 intensity"`.

From 1.35 M raw rows this yields 23,574 zone-month observations. A region
qualifies as a forecast target only with **≥ 36 monthly observations**; the final
**12 months** are held out as ground truth and never seen in training.

**Frequency normalisation.** Ember timestamps are month-start; G3P timestamps are
mid-month (`2002-04-16`). Re-indexing the G3P series onto a month-start
frequency directly discards every observation, so the water index is first
normalised by calendar period and then assigned a monthly frequency, preserving
all 225 observations. G3P also has genuine gaps — 225 observations across a
258-month span (12.8% missing) — which are left as missing rather than filled.

## 2.4 Aligned dataset

The join produces **6,131 sites × 16 columns**: identity (`site_id`, `name`,
`company`, `city`), geography (`latitude`, `longitude`, `country_iso3`,
`country`, `name_1`), water (`pfaf_id`, `water_stress_bws`), grid
(`grid_zone_id`, `electricity_gwh`, `co2_intensity`), climate
(`temperature_c`), and the graph flag `is_node`.

| Property | Value |
|---|---|
| Sites | 6,131 |
| Countries / admin-1 regions | 141 / 487 |
| Distinct grid zones | 96 (95 containing ≥1 graph node) |
| Distinct operators | 2,319 |
| Graph nodes (`is_node`) | **4,889** |
| Forecast targets | 95 zones × 2 resources + 24 basins |

`is_node` is derived, not supplied: it marks the 4,889 sites for which **all four**
node features are present. The 1,242 sites lacking a grid-zone match lose
electricity, carbon and temperature together as one block, which is what
separates node from non-node.

## 2.5 Graph construction

Edges encode a **spatial co-location prior** — two sites are linked when they
share a grid zone or a water basin — capped at `MAX_INTRA_GROUP_K` neighbours
per group to prevent dense zones from dominating. The resulting graph has
**4,889 nodes and 71,165 directed edges**, of which 4,889 are self-loops, leaving
**66,276 real edges** (mean degree ≈ 13.6).

Node features are the four aligned quantities
(`water_stress_bws`, `temperature_c`, `electricity_gwh`, `co2_intensity`).
A two-layer GAT autoencoder (hidden 16 × 4 heads → 8-dim embedding, linear
decoder, MSE reconstruction, 400 epochs, seed 42) is trained unsupervised — no
labels and no fabricated target.

A **random-control graph** with identical node count, edge count and degree
distribution but shuffled endpoints is trained under the same protocol. Because
raw random rewiring also changes the effective blending weight, an
**alpha-calibrated** control is additionally constructed so that mean effective
alpha matches the co-location graph by construction (0.0367 in both), isolating
topology from weight magnitude.

Reconstruction MSE separates the two graphs cleanly — **0.034 (co-location)
versus 0.469 (random)** — confirming the co-location structure is learnable
before any forecasting question is asked.

## 2.6 Known data-quality items

- **Sentinel values.** `pfaf_id` (7 rows) and `water_stress_bws` (12 rows) carry
  `-9999` no-data codes that pass a null check and would corrupt any mean or
  min–max scaling; `water_stress_bws` has a true range of 0–5. None of these rows
  fall in the node set, so model inputs are unaffected, but any statistic over
  the full table must mask them.
- **Free-text fields.** `city` contains 1,719 empty strings and 96 rows with the
  literal value `"States"`, a country-name split artefact. Not used as a feature
  or join key.
- **Temperature is a live snapshot.** Re-running ingestion refetches current
  conditions, so `is_node` membership can shift by a few sites between runs
  independently of any code change. Pin the temperature file for exact
  reproducibility.
