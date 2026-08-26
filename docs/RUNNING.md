# Running the notebook

## Status

Your machine already has everything the pipeline needs except the data:

```
torch 2.10.0+cpu   pandas 3.0.0   numpy 2.4.2   statsmodels 0.14.6
sklearn 1.8.0      scipy 1.17.0   pyarrow 25.0.1  matplotlib 3.10.8
```

`torch_geometric` is **not** needed — it was imported in §1 but never used
(the `.pt` payloads are plain dicts; `torch.load` handles them). That import and
its pip line have been removed.

`openai` is needed **only** for §6B (Nexus). SARIMA and the entire GAT-weighting
and scoring path run without it.

## Step 1 — check what's missing

```bash
python scripts/preflight.py
```

Right now that reports 4 missing inputs. The notebook's §1 raises
`FileNotFoundError` and refuses to substitute a source, so it will not run
until these exist.

| key | what to get | notes |
|---|---|---|
| `ember_us` | Ember monthly electricity — United States | filename must contain `us_monthly` |
| `ember_eu` | Ember monthly electricity — Europe | must contain `europe_monthly` |
| `ember_in` | Ember monthly electricity — India | must contain `india_monthly` |
| `g3p_tws` | G3P total water storage per river basin | must contain `tws_rivbas` |

Matching is by **substring**, case-insensitive, searched recursively under
`data/`. Vendor filenames usually work unchanged — just drop the CSVs in
`data/` and re-run preflight.

`graph_nodes.parquet` appears in §1's path table but is **not** in its
`REQUIRED` list and is never read. You do not need it.

### Where these come from

Both are open datasets, but neither has a stable direct-download URL worth
hardcoding here:

- **Ember** publishes monthly electricity data (generation + CO2 intensity) at
  ember-energy.org. You want the per-region monthly series, not the yearly.
- **G3P** (Global Gravity-based Groundwater Product, g3p.eu) publishes the
  river-basin total-water-storage anomaly file, `G3P_v1.12_tws_rivbas.csv`.

The notebook's own error message says the G3P file came from a sibling notebook
in this project, `arima-xlstm-water-modelling.ipynb` (Part 1) — and that the
`.pt` files came from `gat-colocation-weights-water-energy.ipynb`. If you still
have those Kaggle inputs attached, pulling the CSVs from there is faster than
re-downloading from source.

## Step 2 — run it

Water is the only resource that needs G3P. If you have the three Ember files but
not G3P, you can still get electricity and carbon by removing `"g3p_tws"` from
the `REQUIRED` list in §1 and dropping `"water"` from `RESOURCES` in §0 — §6/§8
then report water as skipped rather than fabricating it.

```bash
export OPENAI_API_KEY="sk-..."      # only if you want §6B Nexus
jupyter lab notebooks/gat_weighted_forecasting.ipynb
```

Run cells top to bottom. Outputs land in `outputs/` (gitignored).

Cost note for §6B: one API call per region-series against `gpt-4o-mini`, capped
by `NEXUS_MAX_LLM_CALLS = 200`, 1.2 s apart. With ~96 grid zones × 2 resources
you are near that cap. Lower it to smoke-test. Without a key, `NEXUS_LLM_OK`
stays False and it falls back to a seasonal-naive forecast — the pipeline still
completes, but "Nexus" then means the fallback, not the LLM.

## What was changed to make this run locally

`scripts/localize_notebook.py` (idempotent, re-runnable) applied 4 edits. All
auto-detect the environment, so **the notebook still runs unmodified on Kaggle**:

| § | before | after |
|---|---|---|
| 0 | `WORK = "/kaggle/working"` | falls back to `../outputs` off-Kaggle |
| 1 | `INPUT_ROOT = "/kaggle/input"` | falls back to `../data` off-Kaggle |
| 1 | `import torch_geometric` | commented out — unused |
| 6B | hardcoded `sk-REPLACE-...` | `os.environ.get("OPENAI_API_KEY", "")` |

The original untouched notebook is still in your Downloads folder.
