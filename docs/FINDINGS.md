# Results — first full run

Real Ember panel, 94 zones, 12-month holdout (~2025-05 → 2026-04).
Reproduce: `python scripts/run_comparison.py --resources electricity,carbon`

## Naming

What ran is the notebook's **Nexus slot** — `MODELS = ["SARIMA", "Nexus"]` — with
the backend swapped from `gpt-4o-mini` to `amazon/chronos-2`. The architecture,
holdout, and scoring are the notebook's. If "Nexus" in the writeup denotes the
LLM specifically, label this `SARIMA vs Chronos-2` instead.

## Pass 1 — Chronos-2 vs SARIMA

| resource | model | RMSE | MAPE | time | wins | t_p | wilcoxon_p |
|---|---|---|---|---|---|---|---|
| electricity | SARIMA | 612.0 | 10.28% | 33.5s | — | — | — |
| electricity | **Chronos-2** | **564.5** | **8.60%** | **2.6s** | 58/94 | 0.119 | **0.022** |
| carbon | SARIMA | 38.3 | 14.14% | 24.8s | — | — | — |
| carbon | **Chronos-2** | **34.1** | **13.26%** | **2.3s** | 63/94 | **0.00069** | **0.001** |
| water (24 basins) | SARIMA | 45.4 | — | 5.1s | — | — | — |
| water (24 basins) | **Chronos-2** | **40.4** | — | **1.1s** | 14/24 | 0.186 | 0.208 |

Water MAPE is omitted deliberately: TWS is an *anomaly* that crosses zero, so
percentage error is undefined/explosive (it printed 171% / 138%). Score water on
RMSE/MAE only.

Chronos-2 has the lower mean error on **all three** resources. It is significant
on carbon only; electricity is median-significant (Wilcoxon) but not mean-
significant; water is not significant at n=24.

**Carbon: Chronos-2 wins, significant on both tests.** ~11% lower RMSE.

**Electricity: split verdict.** Wilcoxon says significant (p=0.022), the paired
t-test does not (p=0.119). Chronos-2 wins on 58 of 94 zones, so it is better
*typically*, but a handful of large misses drag the mean. Report the Wilcoxon —
it is the appropriate test for skewed per-zone error — and state the t-test
alongside it. Do not claim a clean win on electricity.

Chronos-2 ran **~12x faster** overall (4.9s vs 58.3s), zero failures, on CPU.

## Pass 2 — the GAT weighting question

### The result that looks impressive but is not

| comparison | GAT | random | diff | p |
|---|---|---|---|---|
| electricity, Chronos-2 | 1019.4 | 1742.3 | −722.9 | ~0 |
| electricity, SARIMA | 1094.7 | 1774.1 | −679.4 | ~0 |
| carbon, Chronos-2 | 27.04 | 30.17 | −3.14 | 2.6e-131 |
| carbon, SARIMA | 30.75 | 32.69 | −1.94 | 3.3e-44 |

GAT crushes the random control everywhere. **This is a tautology, not evidence.**
See the mechanism section below.

### The result that actually answers the research question

| comparison | GAT | unweighted | diff | p | verdict |
|---|---|---|---|---|---|
| electricity, Chronos-2 | 1019.39 | 1017.04 | **+2.35** | 1.05e-05 | GAT **worse** |
| electricity, SARIMA | 1094.72 | 1092.57 | **+2.15** | 5.88e-05 | GAT **worse** |
| carbon, Chronos-2 | 27.037 | 27.054 | −0.017 | 7.7e-08 | GAT better by **0.06%** |
| carbon, SARIMA | 30.750 | 30.750 | +0.000 | 0.999 | no difference |

**Graph weighting does not improve forecasts.** On electricity it makes them
slightly worse; on carbon the "improvement" is 0.06% of the error — statistically
detectable only because n=4,887, and practically nil. This is the null result the
notebook was built to report honestly, and it now has real data behind it.

## Water breaks the "GAT beats random" story

| comparison | GAT | random | p | verdict |
|---|---|---|---|---|
| water, Chronos-2 | 31.21 | 31.45 | 1.3e-06 | GAT slightly lower |
| water, **SARIMA** | **48.38** | **47.55** | 4.5e-30 | **random BEATS GAT** |

On water with SARIMA the random control is *significantly better* than the
co-location graph. That is the tell: the huge GAT-over-random margins on
electricity were never signal, they were scale. TWS anomalies span roughly
-500..392 mm and are centred near zero for every basin, so blending across
basins cannot blow up the magnitude — and mild cross-region averaging acts as
regularisation, which helps. Electricity levels differ between zones by orders of
magnitude, so the same operation is catastrophic.

## Mechanism — why the design cannot answer the question as posed

Edge composition, measured against both region definitions:

| graph | intra-zone | inter-zone | intra-basin | inter-basin |
|---|---|---|---|---|
| colocation | **98.1%** | 1.9% | **97.7%** | 2.3% |
| random control | 3.9% | **96.1%** | 12.8% | **87.2%** |

The co-location graph is ~98% *within-region* under **both** groupings — it is
very nearly a re-encoding of "same region".

Forecasts are produced **per zone**, so every site in a zone receives an
*identical* 12-month vector. Blending a site with a neighbour in the same zone
therefore changes nothing at all.

Consequence, measured:

| graph | blending is an exact no-op | actually changes the forecast |
|---|---|---|
| colocation | 4,461 sites (**91.3%**) | 421 sites (8.6%) |
| random control | 0 sites (0%) | 4,887 sites (100%) |

So:

- **"GAT beats random" is structurally guaranteed.** Co-location edges preserve
  zone identity, making blending a near-no-op; random edges destroy it, blending
  Virginia's GWh with Delhi's. The comparison measures whether shuffling zone
  labels hurts — it does — and says nothing about co-location signal.
- **The co-location graph can only act through its 1.9% inter-zone edges**, and
  those necessarily import another zone's level, which is why the electricity
  delta is positive (worse).

The graph is at **site** granularity; the forecast is at **zone** granularity.
The finer structure is invisible to the coarser target. More data will not fix
this — the design has to change.

## What would actually test the hypothesis

1. **A control that holds zone structure constant.** Rewire edges *within* zones
   at random. The current control confounds "co-location" with "same zone", so it
   cannot isolate the former. This is the single highest-value fix.
2. **Site-level targets.** The hypothesis is about sites; the data is about
   zones. Without per-site series there is nothing for the graph to explain.
3. **Covariates instead of post-hoc blending.** Chronos-2 accepts covariates and
   has a `cross_learning` flag for related series — feeding neighbour series in
   at inference is a far more direct test than averaging finished forecasts, and
   it is closer to what §5's native-injection demo was reaching for.

## Caveat

`final_reconstruction_mse` (0.033 colocation vs 0.474 random) shows the
co-location graph *is* learnable and carries real structure. That is not in
dispute. What this run shows is that the structure does not transfer to
zone-level forecast accuracy under the current blending design.
