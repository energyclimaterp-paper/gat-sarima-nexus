"""Local forecaster backends for the notebook's `Nexus` slot -- no API key.

Drop-in for §6B. Same contract as the original OpenAI path:

    forecast(series_train: pd.Series, pred_len: int) -> (np.ndarray[pred_len], tag)

Backends
--------
chronos : amazon/chronos-2 (120M, encoder-only TSFM). Purpose-built for
          numeric forecasting; runs on CPU. Default.
llm     : any local chat model via an OpenAI-compatible endpoint (Ollama,
          llama.cpp, LM Studio). Keeps the original prompt/JSON contract so
          "Nexus is an LLM" stays literally true.
naive   : the notebook's own seasonal fallback, for reference.

Both real backends are deterministic given a seed, unlike the hosted API.
"""
from __future__ import annotations
import json, os
import numpy as np
import pandas as pd

_PIPE = None


# --------------------------------------------------------------------- naive
def naive_forecast(series_train: pd.Series, pred_len: int) -> np.ndarray:
    """Identical to the notebook's `_nexus_naive_fallback`: half persistence,
    half month-of-year climatology."""
    mm = series_train.groupby(series_train.index.month).mean()
    last_val = float(series_train.iloc[-1])
    last = series_train.index[-1]
    return np.array([
        0.5 * last_val + 0.5 * mm.get(((last.month - 1 + i + 1) % 12) + 1, series_train.mean())
        for i in range(pred_len)
    ], float)


# ------------------------------------------------------------------- chronos
def load_chronos(model_id: str = "amazon/chronos-2", device: str = "cpu"):
    """Load once and cache. CPU is deliberate: 120M params, and it sidesteps
    the sm_120/Blackwell PyTorch situation entirely."""
    global _PIPE
    if _PIPE is None:
        from chronos import BaseChronosPipeline
        _PIPE = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
    return _PIPE


def chronos_forecast(series_train: pd.Series, pred_len: int) -> np.ndarray:
    """Median of the predicted quantiles -- the point forecast the notebook scores."""
    import torch
    pipe = load_chronos()
    ctx = torch.tensor(series_train.values, dtype=torch.float32)
    q, mean = pipe.predict_quantiles(
        [ctx], prediction_length=pred_len,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    # q[0] is (n_variates, horizon, n_quantiles); univariate -> variate 0, median -> idx 1
    out = np.asarray(q[0][0, :, 1], dtype=float)
    if out.shape[0] != pred_len or not np.isfinite(out).all():
        raise ValueError("chronos returned malformed forecast")
    return out


# ----------------------------------------------------------------------- llm
def llm_forecast(series_train: pd.Series, pred_len: int,
                 base_url: str | None = None, model: str | None = None,
                 hist_points: int = 48) -> np.ndarray:
    """Original prompt, pointed at a local OpenAI-compatible server."""
    from openai import OpenAI
    base_url = base_url or os.environ.get("LOCAL_LLM_URL", "http://localhost:11434/v1")
    model = model or os.environ.get("LOCAL_LLM_MODEL", "qwen3:8b")
    client = OpenAI(base_url=base_url, api_key="not-needed")

    H = [{"m": str(d)[:7], "v": round(float(v), 3)} for d, v in series_train.items()][-hist_points:]
    prompt = ("You are an expert monthly time-series forecaster that internally combines macro-trend, "
              "12-month seasonality, and bias-calibration reasoning in a SINGLE pass. "
              f"History (last {len(H)} months): {json.dumps(H)}. "
              f"Forecast the next {pred_len} monthly values. "
              f"Return ONLY a JSON array of exactly {pred_len} numbers - no prose, no keys.")
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
    )
    txt = resp.choices[0].message.content.strip().strip("`")
    txt = txt[txt.find("["): txt.rfind("]") + 1] if "[" in txt else txt
    arr = np.asarray(json.loads(txt), float)[:pred_len]
    if len(arr) != pred_len or not np.isfinite(arr).all():
        raise ValueError("LLM returned malformed / wrong-length forecast")
    return arr


# -------------------------------------------------------------------- router
def make_nexus_forecast(backend: str = "chronos", **kw):
    """Return a `nexus_forecast(series, pred_len) -> (array, tag)` closure that
    falls back to naive on failure, exactly like the original."""
    fn = {"chronos": chronos_forecast, "llm": llm_forecast,
          "naive": lambda s, n: naive_forecast(s, n)}[backend]

    def nexus_forecast(series_train, pred_len):
        if backend == "naive":
            return naive_forecast(series_train, pred_len), "naive"
        try:
            return fn(series_train, pred_len, **kw), backend
        except Exception:
            return naive_forecast(series_train, pred_len), "naive-fallback"

    return nexus_forecast
