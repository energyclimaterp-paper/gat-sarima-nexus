"""Point Part B's Nexus slot (§6D) at a local backend instead of Gemini.

Part B differs from the v1 notebook in three ways that the injected block must
respect:
  1. the whole §6D body is nested inside `if not BASE_CACHE_HIT:` -> 4-space indent
  2. the tally dict is {"LLM", "stat-fallback"}, not {"LLM", "naive-fallback"}
  3. the call counter is GEMINI_CALLS, not OPENAI_CALLS

A successful backend call reports as "LLM" because that is this section's key for
"the Nexus model produced this, rather than the statistical fallback". The real
backend name is printed so the log is unambiguous.

    python scripts/use_local_backend_v2.py chronos   # default
    python scripts/use_local_backend_v2.py gemini    # revert to the hosted API
"""
import ast, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "v2", "partB_gat_weighted_forecasting.ipynb")
BEG = "    # --- NEXUS LOCAL BACKEND (begin) ---"
END = "    # --- NEXUS LOCAL BACKEND (end) ---"
ANCHOR = "    n_series_total = sum("

backend = (sys.argv[1] if len(sys.argv) > 1 else "chronos").lower()
if backend not in {"chronos", "llm", "gemini"}:
    sys.exit(f"unknown backend {backend!r}; use chronos | llm | gemini")


def strip_block(text):
    return re.sub(re.escape(BEG) + r".*?" + re.escape(END) + r"\n?", "", text, flags=re.S)


block = f'''{BEG}
    import sys as _sys
    _sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
    from nexus_local import make_nexus_forecast
    NEXUS_BACKEND = "{backend}"
    _nexus_impl = make_nexus_forecast(NEXUS_BACKEND)
    NEXUS_LLM_OK = True
    GEMINI_CALLS = 0

    def nexus_forecast(series_train, pred_len):
        """Local backend wrapped in this section's own bookkeeping contract.
        §6D tallies into nexus_src{{"LLM", "stat-fallback"}}; a successful model
        call must report as "LLM" (the model produced it, not the fallback)."""
        global GEMINI_CALLS
        fc, tag = _nexus_impl(series_train, pred_len)
        if tag != "naive-fallback":
            GEMINI_CALLS += 1
            return fc, "LLM"
        return _nexus_statistical(series_train, pred_len), "stat-fallback"

    print(f"Nexus backend: {{NEXUS_BACKEND}} (local, no API key, deterministic) "
          f"-- tallied under the 'LLM' key; 'stat-fallback' still means the "
          f"macro/micro statistical path in this cell")
{END}

'''

nb = json.load(open(NB, encoding="utf-8"))
cell = next(c for c in nb["cells"]
            if c["cell_type"] == "code" and "def nexus_forecast" in "".join(c["source"]))
src = strip_block("".join(cell["source"]))

if backend == "gemini":
    print("Part B §6D reverted to the Gemini path")
else:
    if ANCHOR not in src:
        sys.exit("could not find the §6D injection anchor")
    src = src.replace(ANCHOR, block + ANCHOR, 1)
    print(f"Part B §6D now uses backend={backend!r}")

ast.parse(src)
cell["source"] = src.splitlines(keepends=True)
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"  markers: {src.count(BEG)} begin / {src.count(END)} end | cell parses clean")
