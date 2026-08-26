"""Rewire notebook Section 6B from the OpenAI API to a local backend.

Injects a marker-delimited block that overrides `nexus_forecast`. The
surrounding loop, the fallback accounting, and every downstream section are
untouched, so `base[("Nexus", resource)]` is populated exactly as before.

    python scripts/use_local_backend.py chronos   # default
    python scripts/use_local_backend.py llm       # local OpenAI-compatible server
    python scripts/use_local_backend.py openai    # revert to the hosted API
"""
import ast, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "gat_weighted_forecasting.ipynb")
BEG = "# --- NEXUS LOCAL BACKEND (begin) ---"
END = "# --- NEXUS LOCAL BACKEND (end) ---"
ANCHOR = "n_series_total = sum("

backend = (sys.argv[1] if len(sys.argv) > 1 else "chronos").lower()
if backend not in {"chronos", "llm", "openai"}:
    sys.exit(f"unknown backend {backend!r}; use chronos | llm | openai")


def strip_block(text):
    """Remove BEG..END inclusive. Marker-delimited, so repeated runs cannot
    leave residue the way line-counting did."""
    return re.sub(re.escape(BEG) + r".*?" + re.escape(END) + r"\n?", "", text, flags=re.S)


block = f'''{BEG}
import sys as _sys
_sys.path.insert(0, os.path.join("..", "src"))
from nexus_local import make_nexus_forecast
NEXUS_BACKEND = "{backend}"
_nexus_impl = make_nexus_forecast(NEXUS_BACKEND)
NEXUS_LLM_OK = True
OPENAI_CALLS = 0

def nexus_forecast(series_train, pred_len):
    """Wraps the local backend in the notebook's own bookkeeping contract.
    §6B counts into nexus_src{{"LLM", "naive-fallback"}} and warns when
    OPENAI_CALLS == 0, so a successful model call must report as "LLM"
    (meaning: the Nexus model produced this, not the fallback)."""
    global OPENAI_CALLS
    fc, tag = _nexus_impl(series_train, pred_len)
    if tag != "naive-fallback":
        OPENAI_CALLS += 1
        return fc, "LLM"
    return fc, "naive-fallback"

print(f"Nexus backend: {{NEXUS_BACKEND}} (local, no API key, deterministic) "
      f"-- counted under the 'LLM' key in this section's tallies")
{END}

'''

nb = json.load(open(NB, encoding="utf-8"))
cell = next(c for c in nb["cells"]
            if c["cell_type"] == "code" and "def nexus_forecast" in "".join(c["source"]))
src = strip_block("".join(cell["source"]))

if backend == "openai":
    print("Section 6B reverted to the OpenAI path")
else:
    if ANCHOR not in src:
        sys.exit("could not find the injection anchor in Section 6B")
    src = src.replace(ANCHOR, block + ANCHOR, 1)
    print(f"Section 6B now uses backend={backend!r}")

ast.parse(src)
cell["source"] = src.splitlines(keepends=True)
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"  markers: {src.count(BEG)} begin / {src.count(END)} end | cell parses clean")
