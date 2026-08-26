# Results — Part A/B v2

Regenerated graph (Part A) + four-model forecasting and region-level evaluation
(Part B). Both executed end to end; Part B's only error is cell 3, a bare
`torch.cuda.get_device_name(0)` diagnostic that fails on CPU-only torch.

Nexus backend = `amazon/chronos-2`, local, no API key.

## 1. Four-model comparison (region-level RMSE, 12-month holdout)

| resource | n | SARIMA | xLSTM | TimesFM | Nexus (Chronos-2) | best |
|---|---|---|---|---|---|---|
| carbon | 95 | 38.01 | 42.32 | 35.50 | **33.85** | Nexus |
| electricity | 95 | 606.05 | 745.31 | 579.64 | **558.88** | Nexus |
| water | 24 | **37.49** | 55.17 | 47.00 | 40.42 | **SARIMA** |

Paired Wilcoxon against SARIMA:

| resource | xLSTM | TimesFM | Nexus |
|---|---|---|---|
| carbon | +11.4% (p=0.017) | −6.6% (p=0.172) | **−10.9% (p=0.00087)** |
| electricity | +23.0% (p=1.4e-05) | −4.4% (p=0.140) | **−7.8% (p=0.020)** |
| water | +47.2% (p=0.00015) | +25.4% (p=0.012) | +7.8% (p=0.584) |

**Nexus wins the two grid resources with significance.** On water, classical
SARIMA beats every learned model — xLSTM and TimesFM significantly worse,
Nexus not significantly different. Foundation models do not win universally, and
the resource where they lose is the one with the shortest history (225 months,
12.8% missing) and an anomaly target centred on zero.

xLSTM is the weakest model on both grid resources, materially worse than the
classical baseline.

## 2. Graph weighting

Region-level, FDR-corrected — N = 95 zones / 24 basins, the correct inferential
unit (site-level N = 4,889 is pseudo-replication and the notebook labels it
non-inferential).

| resource | gat_weighted | unweighted | random (raw) | random (calibrated) |
|---|---|---|---|---|
| carbon | 33.247 | 33.213 | 35.903 | 33.794 |
| electricity | 580.163 | 562.194 | 1076.199 | 774.183 |
| water | 39.550 | 39.518 | 39.538 | 39.370 |

| comparison | carbon | electricity | water |
|---|---|---|---|
| GAT vs unweighted | p=0.349 | p=0.357 | p=0.165 |
| GAT vs random (raw) | p=0.0009 | p<1e-5 | p=0.922 |
| GAT vs random (**calibrated**) | p=0.333 | **p<1e-5** | p=0.614 |

**Graph weighting does not beat not weighting on any resource.** The mixed-effects
model agrees (gat coefficient p = 0.948 / 0.612 / 0.829).

The alpha calibration matters: raw random control had mean effective alpha
0.0688 against co-location's 0.0367, nearly 2×. Once matched by construction,
carbon's apparent advantage over random disappears and only **electricity**
survives — and even there the comparison still confounds co-location with
"same region", since random rewiring also crosses region boundaries.

Co-location itself is real: same-group correlation 0.41 vs 0.07 across groups for
electricity (Mann-Whitney p≈0). The signal exists; it does not convert into
forecast gains through this blending scheme.

## 3. Headroom

| resource | oracle best-single | graph_weighted | simple_average | random |
|---|---|---|---|---|
| carbon | **24.86** | 26.47 | 26.48 | 28.41 |
| electricity | **866.56** | 1002.50 | 999.80 | 1463.16 |
| water | **26.97** | 32.25 | 32.25 | 32.22 |

Per-site model selection remains worth far more than edge weighting: 6% on
carbon, 13% on electricity, 16% on water.

## 4. Bug found and fixed

Part B's `[1.4]` frequency fix applied `train_series.asfreq("MS")` inside
`fit_sarima`. Ember timestamps are month-start so this is a no-op there, but
**G3P water timestamps are mid-month** (`2002-04-16`), so re-indexing onto
month-start produced a **100% NaN series**. SARIMAX does not raise — the Kalman
filter treats NaN as missing — so it silently fit nothing and emitted a
degenerate forecast.

Effect on SARIMA water RMSE:

| basin | before | after |
|---|---|---|
| Ganges | 178.8 | 58.3 |
| Indus | 161.6 | 38.4 |
| Danube | 108.9 | 20.1 |
| Rhine | 103.1 | 26.0 |
| **mean over 24 basins** | **103.02** | **37.49** |

The repair (`scripts/fix_freq_bug.py`) normalises the index by calendar period
before applying the frequency, preserving all 225 observations.

**This inverted the water conclusion.** Pre-fix, Nexus appeared to beat SARIMA on
water by 60.8% (p=2.4e-07); post-fix SARIMA is the best water model and Nexus is
7.8% worse and not significant. It also removed water's apparent significance
against the calibrated random control (p=0.045 → 0.614). Any water number from
the pre-fix run should be discarded.
