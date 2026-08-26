# Results record

Source: `notebooks/gat_weighted_forecasting.ipynb` executed end to end (0 cell errors). Nexus backend = **chronos-2** (local, no API key).

Holdout = 12 months. Regions: electricity/carbon = grid zone, water = river basin.

## 1. Coverage

| model | resource | forecast_rows |
|---|---|---|
| Nexus | carbon | 1128 |
| Nexus | electricity | 1128 |
| Nexus | water | 288 |
| SARIMA | carbon | 1128 |
| SARIMA | electricity | 1128 |
| SARIMA | water | 288 |

## 2. SARIMA vs Nexus — per-region paired

| resource | n_regions | SARIMA_RMSE | Nexus_RMSE | delta | pct_change | nexus_wins | t_p | wilcoxon_p |
|---|---|---|---|---|---|---|---|---|
| carbon | 94 | 38.291 | 34.109 | -4.182 | -10.92 | 63 | 0.0006871 | 0.001001 |
| electricity | 94 | 612.001 | 564.54 | -47.461 | -7.76 | 58 | 0.1185 | 0.02197 |
| water | 24 | 45.444 | 40.42 | -5.024 | -11.06 | 14 | 0.186 | 0.2076 |

Negative `delta` = Nexus lower RMSE (better).

## 3. Per-site scoring after GAT blending (§10)

| resource | condition | MAPE | RMSE | MAE |
|---|---|---|---|---|
| carbon | gat_weighted | 10.5615 | 26.6704 | 22.5317 |
| carbon | random_control | 13.8173 | 29.5712 | 25.2698 |
| carbon | unweighted | 10.5755 | 26.6764 | 22.5359 |
| electricity | gat_weighted | 6.382 | 1011.1664 | 857.616 |
| electricity | random_control | 39.9222 | 1728.5237 | 1589.7056 |
| electricity | unweighted | 6.0417 | 1008.7838 | 855.2728 |
| water | gat_weighted | 166.0429 | 35.0868 | 28.583 |
| water | random_control | 168.1847 | 34.8504 | 28.4505 |
| water | unweighted | 166.0698 | 35.1006 | 28.5971 |

## 4. Significance: gat_weighted vs baselines (§10)

| resource | comparison | n | mean_diff_rmse | t_stat | t_p | wilcoxon_stat | wilcoxon_p | note |
|---|---|---|---|---|---|---|---|---|
| carbon | gat_vs_unweighted | 4887 | -0.0060061036954892 | -1.4468968621685991 | 0.1479899619900512 | 1574257.5 | 1.7542107435387948e-07 | mean_diff<0 => GAT lower RMSE (better) |
| carbon | gat_vs_random_control | 4887 | -2.900784904679301 | -23.2002441507246 | 4.780613054001496e-113 | 3705491.0 | 7.839211753635347e-117 | mean_diff<0 => GAT lower RMSE (better) |
| electricity | gat_vs_unweighted | 4887 | 2.3825371153444768 | 4.4094196739337645 | 1.0587313528463922e-05 | 2430517.5 | 7.157830404364802e-06 | mean_diff<0 => GAT lower RMSE (better) |
| electricity | gat_vs_random_control | 4887 | -717.357331045941 | -44.26693741746878 | 0.0 | 1460649.0 | 0.0 | mean_diff<0 => GAT lower RMSE (better) |
| water | gat_vs_unweighted | 4887 | -0.0138091220279377 | -4.747784705614787 | 2.1153521382668445e-06 | 354065.5 | 7.464663161120478e-57 | mean_diff<0 => GAT lower RMSE (better) |
| water | gat_vs_random_control | 4887 | 0.2363616852598727 | 3.853656512597917 | 0.0001178595472924 | 4910471.0 | 6.630593344899387e-27 | mean_diff<0 => GAT lower RMSE (better) |

## 5. Version comparison incl. oracle (§10)

| version | resource | RMSE | MAE |
|---|---|---|---|
| best_single_model(ORACLE) | carbon | 26.0176 | 21.4449 |
| best_single_model(ORACLE) | electricity | 892.2789 | 751.361 |
| best_single_model(ORACLE) | water | 29.2456 | 23.0302 |
| simple_average | carbon | 26.6764 | 22.5359 |
| simple_average | electricity | 1008.7838 | 855.2728 |
| simple_average | water | 35.1006 | 28.5971 |
| graph_weighted | carbon | 26.6704 | 22.5317 |
| graph_weighted | electricity | 1011.1664 | 857.616 |
| graph_weighted | water | 35.0868 | 28.583 |
| random_edge_check | carbon | 29.5712 | 25.2698 |
| random_edge_check | electricity | 1728.5237 | 1589.7056 |
| random_edge_check | water | 34.8504 | 28.4505 |

## 6. Run configuration

```json
{
  "forecast_unit": {
    "electricity": "grid_zone",
    "carbon": "grid_zone",
    "water": "basin (nearest G3P river-basin centroid, cutoff 2500 km)"
  },
  "resources_forecast": [
    "electricity",
    "carbon",
    "water"
  ],
  "resources_skipped": {
    "water_stress": "static per basin (Aqueduct bws_score has no time axis)",
    "temperature": "live Open-Meteo per-site SNAPSHOT (a 4th GAT node feature in Step 1-3), but a single current reading with NO monthly history => cannot be forecast as a series; it influences Step 4 INDIRECTLY through the GAT weights"
  },
  "models": [
    "SARIMA",
    "Nexus"
  ],
  "model_run_order_rationale": "SARIMA baseline, then Nexus (live LLM calls)",
  "weight_field_used": "site_attention_received (min-max scaled, x ALPHA_MAX)",
  "ALPHA_MAX": 0.5,
  "changed_fraction_gat": 0.09030761885273855,
  "changed_fraction_random": 0.9997953754859832
}
```

## 7. Files in this directory

- `RESULTS.md` — 0.0 KB
- `compare_base.csv` — 0.2 KB
- `compare_sites.csv` — 1452.9 KB
- `forecast_failures_log.csv` — 0.0 KB
- `forecasts_gat_weighted.csv` — 17129.9 KB
- `forecasts_random_control_weighted.csv` — 17292.8 KB
- `forecasts_unweighted.csv` — 17085.2 KB
- `per_region_base_forecasts.csv` — 277.3 KB
- `step5_aggregate_by_resource.csv` — 0.4 KB
- `step5_diagram_versions.csv` — 0.5 KB
- `step5_metrics_site_resource.csv` — 2565.7 KB
- `step5_significance.csv` — 1.0 KB
- `summary_model_comparison.csv` — 0.3 KB
- `weight_injection_log.json` — 1.2 KB
