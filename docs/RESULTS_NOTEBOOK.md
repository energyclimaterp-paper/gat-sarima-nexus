# Notebook run — SARIMA vs Nexus (Chronos-2)

`notebooks/gat_weighted_forecasting.ipynb`, executed end to end via nbconvert.
Executed copy: `notebooks/executed_run.ipynb`. **0 cells errored**; both models
covered 212/212 region-series with 0 failures.

Nexus backend = `amazon/chronos-2`, local, no API key. The upstream five-agent
notebook was excluded per instruction.

## Head-to-head (per region, paired)

| resource | n | SARIMA | **Nexus** | delta | Nexus wins | t_p | wilcoxon_p |
|---|---|---|---|---|---|---|---|
| carbon | 94 | 38.29 | **34.11** | −4.18 (−10.9%) | 63/94 | **0.00069** | **0.001** |
| electricity | 94 | 612.00 | **564.54** | −47.46 (−7.8%) | 58/94 | 0.119 | **0.022** |
| water | 24 | 45.44 | **40.42** | −5.02 (−11.0%) | 14/24 | 0.186 | 0.208 |

**Nexus beats SARIMA on all three resources.** Significant on carbon (both
tests); electricity is median-significant (Wilcoxon) but not mean-significant;
water is directionally better but n=24 is underpowered.

## Ensemble effect

§10 averages the models, so adding Nexus alongside SARIMA moves the unweighted
baseline. Comparing the SARIMA-only run against the two-model run:

| resource | SARIMA only | + Nexus | change |
|---|---|---|---|
| electricity | 1092.57 | **1008.78** | −7.7% |
| carbon | 30.75 | **26.68** | −13.2% |
| water | 48.40 | **35.10** | −27.5% |

This is the strongest result in the run: **Nexus earns its place in the ensemble
on every resource**, and most dramatically on water.

## GAT weighting — the notebook's own verdict (§37)

```
carbon     : no significant improvement (mean_diff_rmse=-0.0060, t_p=0.148,    n=4887)
electricity: no significant improvement (mean_diff_rmse=+2.3825, t_p=1.06e-05, n=4887)
water      : SIGNIFICANT improvement    (mean_diff_rmse=-0.0138, t_p=2.12e-06, n=4887)
```

Water's "significant improvement" is **0.0138 RMSE on a base of 35.10 — a 0.04%
effect**, detectable only because n=4,887. Electricity is significantly *worse*.
The conclusion is unchanged from the earlier harness run: graph weighting does
not meaningfully improve forecasts.

The random-edge check still inverts on water: `gat 35.0868` vs `random 34.8504`
— the random control is *better*, which remains the cleanest evidence that the
large GAT-over-random margins elsewhere are a magnitude artifact, not signal.

## The oracle line is the useful one (§35)

| resource | best-single (ORACLE) | graph_weighted | random | simple_average |
|---|---|---|---|---|
| carbon | **26.02** | 26.67 | 29.57 | 26.68 |
| electricity | **892.28** | 1011.17 | 1728.52 | 1008.78 |
| water | **29.25** | 35.09 | 34.85 | 35.10 |

Picking the better model *per site* beats graph weighting by far more than graph
weighting beats the plain average — 12% on electricity, 17% on water. **Per-site
model selection is where the headroom is**, not edge weighting.

## Cross-validation

The notebook and the standalone harness (`scripts/run_comparison.py`) were
written independently and agree to 4 decimal places on the SARIMA-only run
(electricity gat_weighted 1094.7172 vs 1094.72; carbon 30.7509 vs 30.75; water
48.3757 vs 48.38), and `changed_fraction_gat = 0.0903` matches the 9%
blending-is-a-no-op figure measured directly from the edge list.
