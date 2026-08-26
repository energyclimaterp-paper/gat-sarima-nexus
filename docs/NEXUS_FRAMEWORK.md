# What Nexus actually is

**Correction.** Earlier in this project I treated "Nexus" as a single model slot
and reported a `SARIMA vs Chronos-2` run as "Nexus vs SARIMA". That was wrong.
Nexus is a **five-agent framework** — "Nexus: An Agentic Framework for Time
Series Forecasting" — implemented in `notebooks/upstream/nexus-ember-in-us-eu.ipynb`
and `notebooks/upstream/nexus-water-modelling.ipynb`.

## The five agents

| agent | symbol | does | non-LLM fallback |
|---|---|---|---|
| ContextAgent | A_ctx | structures history: seasonal labels, energy-mix tiers, per-region event flags -> `H` | rule-based (always) |
| MacroAgent | A_macro | coarse full-horizon trajectory `X_macro` + narrative `R_macro` | **SARIMA(1,1,1)(1,1,0,12)** |
| MicroAgent | A_micro | per-step forecast `X_micro` + traces `R_micro`, conditioned on macro | seasonal-blend (month-of-year climatology) |
| CalibAgent | A_calib | 6-fold walk-forward -> guidelines `G` (bias, macro/micro weights), **validation-gated** | pure statistics (no LLM) |
| SynthAgent | A_syn | `final = w_macro*X_macro + w_micro*X_micro - bias` | pure arithmetic (no LLM) |

Constants: `N_SPLITS = 6`, `MIN_IMPROV = 0.05`. Bias correction is applied only
if it clears 5% improvement on the held-out validation fold, else zeroed.

`CalibAgent` derives weights by inverse-RMSE over the folds:
`macro_w = 1 - macro_rmse/(macro_rmse+micro_rmse)`, normalised — so the agent
that backtests better gets more weight, per region, independently.

## It runs without an API key

`USE_LLM_AGENTS` gates the LLM path. With no key it is `False` and every agent
falls through to its deterministic fallback. The Ember notebook wires Gemini
(`google-generativeai`); the documentation table also names Claude Sonnet 4.6 for
MacroAgent. **So Nexus was never blocked on an API key** — it degrades to a
statistical ensemble.

## Two consequences that matter for the writeup

**1. "Nexus vs SARIMA" is not model-vs-model. Nexus *contains* SARIMA.**
MacroAgent's fallback is SARIMA, and CalibAgent fits SARIMA in every one of its 6
folds. In no-LLM mode Nexus is essentially *a calibrated blend of SARIMA and
seasonal climatology*. Beating SARIMA is then an ensemble-beats-its-own-component
result — much weaker than it sounds, and a reviewer will catch it. State the mode
(LLM or fallback) whenever a Nexus number is reported.

**2. The GAT notebook contains a collapsed Nexus, not the real one.**
`gat_weighted_forecasting.ipynb` §6B is headed "MINIMISED OpenAI: exactly 1 call
per region-series", and its single prompt asks the model to "internally combine
macro-trend, 12-month seasonality, and bias-calibration reasoning in a SINGLE
pass". Those three named components map exactly onto MacroAgent, MicroAgent and
CalibAgent. It is a deliberate cost-reduced flattening of the five-agent
pipeline into one call — not the framework itself.

## What my earlier run therefore measured

`scripts/run_comparison.py` compared **SARIMA vs Chronos-2 as base forecasters**,
inside the GAT notebook's blending pipeline. That comparison is valid on its own
terms and its numbers stand (see `FINDINGS.md`), but it is **not** a Nexus
evaluation. Relabel it `SARIMA vs Chronos-2`.

## Where Chronos-2 actually belongs

As the **MacroAgent backend**, replacing the SARIMA fallback. That gives a clean
ablation holding the rest of the framework fixed:

| variant | MacroAgent | MicroAgent |
|---|---|---|
| Nexus (baseline, no LLM) | SARIMA | seasonal blend |
| **Nexus (Chronos)** | **chronos-2** | seasonal blend |
| Nexus (LLM) | Gemini / Claude | LLM |

CalibAgent then *learns* whether to trust the new macro agent — its inverse-RMSE
weighting is exactly the right mechanism to arbitrate this, and it does so per
region without hand-tuning. This is a far better use of the earlier benchmark
result (chronos-2 beat SARIMA on all three resources) than swapping the whole
slot.
