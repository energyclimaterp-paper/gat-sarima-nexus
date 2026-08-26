# Nexus — GAT-weighted resource forecasting

Does a **spatial co-location graph** over data-centre sites carry signal that improves
monthly resource forecasts (electricity, carbon, water) beyond what the regional time
series already gives you?

The test is deliberately falsifiable. A GAT trained on real co-location edges is scored
against an identically-shaped GAT trained on random edges — and, because raw random
rewiring also changes the blending weight, against an **alpha-calibrated** control whose
mean effective alpha is matched by construction. Inference is at the **region** level
(95 grid zones, 24 river basins), not per site, because sites within a region share one
forecast and site-level N is pseudo-replication.

**Headline: the graph does not improve forecasts.** Chronos-2, in the Nexus slot, does.

## Results

Region-level RMSE, 12-month holdout — see [docs/FINDINGS_V2.md](docs/FINDINGS_V2.md).

| resource | n | SARIMA | xLSTM | TimesFM | Nexus (Chronos-2) |
|---|---|---|---|---|---|
| carbon | 95 | 38.01 | 42.32 | 35.50 | **33.85** |
| electricity | 95 | 606.05 | 745.31 | 579.64 | **558.88** |
| water | 24 | **37.49** | 55.17 | 47.00 | 40.42 |

Nexus wins both grid resources with significance (carbon p=0.0009, electricity p=0.020).
On water, classical SARIMA beats every learned model.

Graph weighting versus no weighting is **null on all three resources**
(FDR-corrected p = 0.349 / 0.357 / 0.165; mixed-effects p = 0.948 / 0.612 / 0.829).
Co-location itself is real — same-region correlation 0.41 against 0.07 across regions —
it simply does not transfer to forecast accuracy through this blending scheme.

## Layout

```
notebooks/v2/     partA — builds the graph and trains the GAT
                  partB — 4 models, blending, region-level evaluation   <- current
notebooks/        v1 pipeline, kept for provenance
notebooks/upstream/  the original five-agent Nexus notebooks (reference)
outputs_v2/partA/ regenerated graph + GAT weights
outputs_v2/partB/ forecasts, metrics, significance tests
outputs/          v1 results
archive_v1/       the originally-supplied GAT weights
scripts/          setup, patching and analysis helpers
docs/             data dictionary, run guide, findings
```

## Getting the data

Raw inputs are **not** committed — 646 MB, and several files exceed GitHub's 100 MB
limit. After cloning:

1. Obtain the sources listed in [docs/DATASET_AND_PREPROCESSING.md](docs/DATASET_AND_PREPROCESSING.md)
   (Ember monthly US/Europe/India, G3P `tws_rivbas`, WRI Aqueduct 4.0 GDB) and place
   them under `data/ember/`, `data/g3p/`, `data/aqueduct/`.
2. Restore the derived artifacts Part B reads:
   ```bash
   mkdir -p data/weights
   cp outputs_v2/partA/gat_weights_*.pt          data/weights/
   cp outputs_v2/partA/graph_nodes.parquet       data/
   cp outputs_v2/partA/checkpoints/aligned_dataset__v1.parquet data/aligned_dataset.parquet
   ```
   (`data/` holds byte-identical copies of Part A's output, so only one copy is tracked.)
3. Check everything resolves:
   ```bash
   python scripts/preflight.py
   ```

## Running

```bash
python scripts/localize_v2.py                  # local paths + one upstream repair
python scripts/use_local_backend_v2.py chronos # Nexus slot -> amazon/chronos-2, no API key
jupyter lab notebooks/v2/partB_gat_weighted_forecasting.ipynb
```

All four models run on CPU. TimesFM loads in ~166 s and forecasts in ~320 ms/series;
Chronos-2 needs no GPU, which sidesteps the Blackwell/sm_120 PyTorch situation entirely.
Full guidance in [docs/RUNNING.md](docs/RUNNING.md) and
[docs/MODEL_CHOICE.md](docs/MODEL_CHOICE.md).

## Documentation

| file | contents |
|---|---|
| [docs/FINDINGS_V2.md](docs/FINDINGS_V2.md) | current results, and the frequency bug that changed the water conclusion |
| [docs/DATASET_AND_PREPROCESSING.md](docs/DATASET_AND_PREPROCESSING.md) | sources, alignment, graph construction, data-quality items |
| [docs/IO_CONTRACT.md](docs/IO_CONTRACT.md) | exactly which inputs each notebook reads and which outputs it writes |
| [docs/NEXUS_FRAMEWORK.md](docs/NEXUS_FRAMEWORK.md) | what Nexus is (five agents) and how the slot is backed here |
| [docs/MODEL_CHOICE.md](docs/MODEL_CHOICE.md) | why Chronos-2, benchmarked locally against the alternatives |
| [docs/RESULTS_NOTEBOOK.md](docs/RESULTS_NOTEBOOK.md) | the v1 run, kept for comparison |
