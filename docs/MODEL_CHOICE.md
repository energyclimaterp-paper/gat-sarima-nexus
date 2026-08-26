# Choosing the local forecaster

## The task, precisely

`nexus_forecast(series_train, pred_len) -> np.ndarray[pred_len]`

The original §6B sends the last 48 monthly values as JSON and asks for 12 numbers
back. There is **no language in this task** — no instructions to follow, no text
to reason over, just numeric context in and a numeric horizon out. That single
fact drives the whole model decision.

A general chat LLM is a poor fit here: it must serialize floats to tokens,
"reason" about them in text, and emit parseable JSON — and it can fail at the
last step, which is why the original code has retry + malformed-output handling.
Time-series foundation models (TSFMs) take tensors and return tensors. No
parsing, no malformed output, deterministic given a seed.

## Candidates considered

| model | params | local? | verdict |
|---|---|---|---|
| **amazon/chronos-2** | 120M | yes | **chosen** — SOTA zero-shot, CPU-fast, covariate support |
| amazon/chronos-bolt-base | 205M | yes | strong, univariate only |
| amazon/chronos-bolt-small | 48M | yes | faster, slightly weaker |
| amazon/chronos-t5-small | 46M | yes | superseded by Bolt |
| google/timesfm-2.5 | 200–500M | yes | competitive; extra dep, no covariates in the same way |
| ibm-granite TTM r2 | ~1–5M | yes | tiny and fast, but fixed context/horizon combos |
| Salesforce Moirai / Toto / TiRex | 100M–1B | yes | leaderboard-competitive; heavier setup |
| Qwen3-8B / Llama-3.1-8B (Q4) | 8B | yes | only if "Nexus must be an LLM" — see below |
| Nixtla TimeGPT | — | **no** | API only, defeats the purpose |

Leaderboard caveat worth keeping in mind: GIFT-Eval rank does not transfer
cleanly to your data. Published rankings put Toto-2.0 and TimesFM-2.5 at or near
the top, but on any single dataset the ordering routinely scrambles. That is why
the numbers below were measured on this machine rather than copied from a table.

## Measured on this machine (CPU, 24 cores)

24 synthetic monthly series, 96 months, 12-month holdout — same shape as the
real targets. Run it yourself: `python scripts/bench_models.py`.

Full head-to-head, all four Chronos-family models actually downloaded and run
(`scripts/bench_models.py`, log in `bench.log`):

| model | params | mean RMSE | vs naive | ms/series | fail |
|---|---|---|---|---|---|
| **chronos-2** | **120M** | **622.7** | **0.29x** | 32.6 | 0 |
| chronos-bolt-base | 205M | 757.0 | 0.36x | 40.2 | 0 |
| chronos-bolt-small | 48M | 761.6 | 0.36x | 15.9 | 0 |
| SARIMA (classical) | - | 783.7 | 0.37x | 80.3 | 0 |
| chronos-t5-small | 46M | 1056.3 | 0.50x | 186.9 | 0 |
| seasonal-naive | - | 2113.3 | 1.00x | 0.4 | 0 |

Chronos-2 beat SARIMA by **~21% RMSE** at **~2.5x the speed**, on CPU, 0 failures.

Three things in that table are worth more than the winner:

1. **Within-family spread is larger than the gap to SARIMA.** chronos-t5-small
   is *worse* than SARIMA (1056 vs 784); the Bolt pair only ties it. "Use a
   foundation model" is not the lesson -- "use the right one" is. Picking by
   family name rather than by measurement would have made the pipeline worse.
2. **Bigger is not better here.** chronos-bolt-small (48M) matched
   chronos-bolt-base (205M) at 2.5x the speed. Parameter count bought nothing.
3. **Newer generation beat more parameters.** chronos-2 at 120M beat bolt-base
   at 205M by 18%.

Load times in the log (200.9s for bolt-base) are one-time weight downloads, not
inference cost. Cached under `~/.cache/huggingface`; chronos-2 is 456 MB.

**Caveat, stated plainly:** these are synthetic series built from trend +
12-month seasonality + Gaussian noise -- a structure both SARIMA and a TSFM
handle well, and it flatters both against the naive baseline. Treat the ordering
as indicative, not as a result. The real comparison happens when the Ember/G3P
data lands, and the notebook already scores it properly. Given point 1 above, it
is worth re-running this sweep on the real series before committing.

## Why this settles PC vs Kaggle

Chronos-2 is 120M parameters and runs on **CPU**. It never touches the GPU, so
the 8GB VRAM ceiling and the Blackwell/sm_120 PyTorch problem both become
irrelevant. The whole Nexus pass projects to well under a minute.

Your RTX 5060 is compute capability **12.0** (Blackwell). Stable PyTorch does
not ship sm_120 kernels; you would need a nightly cu128/cu129 build, and JIT
compilation is still broken there (missing `libnvptxcompiler.so`), which breaks
FlashAttention and custom kernels. Avoiding that path entirely is a feature.

## If Nexus must stay an LLM

If the writeup's claim is specifically "an LLM forecaster," swapping to a TSFM
changes what the experiment measures — that is a real integrity consideration,
not a technicality. In that case use `backend="llm"` in `src/nexus_local.py`,
which keeps the original prompt and JSON contract verbatim and points at any
OpenAI-compatible local server:

```bash
ollama serve && ollama pull qwen3:8b
export LOCAL_LLM_URL=http://localhost:11434/v1
export LOCAL_LLM_MODEL=qwen3:8b
```

Ollama ships its own CUDA build and handles Blackwell better than a torch wheel.
A 7–8B model at Q4_K_M is ~5GB and fits your 8GB. Expect it to be slower and
less accurate at this task than a 120M TSFM — that is the trade you are making
to keep the "LLM" label.

Cleanest option: run **both**. Add Chronos as a third model alongside SARIMA and
the LLM. `MODELS` in §0 is just a list, and the scoring in §10 already
generalizes over it.
