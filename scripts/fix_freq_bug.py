"""Repair the [1.4] frequency fix in Part B's fit_sarima.

The original line is:

    train_series = train_series.asfreq("MS") if train_series.index.freq is None else train_series

`asfreq("MS")` REINDEXES onto month-start timestamps. Ember dates are already
month-start so it is a harmless no-op there. G3P water dates are mid-month
(2002-04-16), so none of them align to month-start and every value becomes NaN
-- SARIMA then fits a 100% NaN series for water and emits a degenerate forecast.

The repair normalises the index to month-start by PERIOD first, which preserves
every observation, then applies asfreq.
"""
import ast, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(ROOT, "notebooks", "v2", "partB_gat_weighted_forecasting.ipynb")

OLD = '    train_series = train_series.asfreq("MS") if train_series.index.freq is None else train_series'
NEW = '''    # [1.4-fix] asfreq("MS") REINDEXES to month-start. Ember is already month-start
    # (no-op), but G3P water is mid-month (2002-04-16) so every value became NaN and
    # SARIMA fit an all-NaN series. Normalise by period first to preserve the values.
    if train_series.index.freq is None:
        _idx = train_series.index
        if not (_idx.day == 1).all():
            train_series = train_series.copy()
            train_series.index = _idx.to_period("M").to_timestamp()
            train_series = train_series[~train_series.index.duplicated(keep="first")]
        train_series = train_series.asfreq("MS")'''

nb = json.load(open(NB, encoding="utf-8"))
hit = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    if "[1.4-fix]" in src:
        print("already repaired"); sys.exit(0)
    if OLD in src:
        src = src.replace(OLD, NEW)
        ast.parse(src)
        cell["source"] = src.splitlines(keepends=True)
        hit += 1

if not hit:
    sys.exit("could not find the [1.4] line to repair")
json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"repaired fit_sarima in {hit} cell(s)")
