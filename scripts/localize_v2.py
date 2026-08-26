"""Point the v2 notebooks at local paths and outputs_v2/. Idempotent.

Same auto-detect approach as scripts/localize_notebook.py: every edit keeps
working on Kaggle, so the notebooks stay portable.

Also repairs one genuine defect in Part A: `section()` is called in cells 3, 18,
21, 27 and 33 but is never defined anywhere in the notebook, so a clean run dies
at the first call and cascades NameErrors through every later cell.
"""
import ast
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# notebooks live in notebooks/v2/, so WORK is two levels up, not one
NBS = {
    "partA_gat_colocation.ipynb":           "../../outputs_v2/partA",
    "partB_gat_weighted_forecasting.ipynb": "../../outputs_v2/partB",
}

HELPER_MARK = "[local-run repair]"
HELPER = (
    '\n\n# ' + HELPER_MARK + ' `section()` is called in this notebook but never defined in it.\n'
    'def section(t):\n'
    '    print("\\n" + "=" * 78 + f"\\n{t}\\n" + "=" * 78)\n'
)

for name, work in NBS.items():
    path = os.path.join(ROOT, "notebooks", "v2", name)
    nb = json.load(open(path, encoding="utf-8"))
    changed = 0

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        orig = src

        if 'WORK = "/kaggle/working"' in src and 'os.path.isdir("/kaggle")' not in src:
            src = src.replace(
                'WORK = "/kaggle/working"',
                'WORK = "/kaggle/working" if os.path.isdir("/kaggle") '
                'else os.path.abspath("' + work + '")')

        if 'INPUT_ROOT = "/kaggle/input"' in src and 'os.path.abspath("../../data")' not in src:
            src = src.replace(
                'INPUT_ROOT = "/kaggle/input"',
                'INPUT_ROOT = "/kaggle/input" if os.path.isdir("/kaggle/input") '
                'else os.path.abspath("../../data")')

        # define the missing helper at the end of the main import cell
        if ("import torch_geometric" in src
                and "def section(" not in src
                and HELPER_MARK not in src):
            src = src.rstrip("\n") + HELPER

        if src != orig:
            ast.parse(src)
            cell["source"] = src.splitlines(keepends=True)
            changed += 1

    if changed:
        json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

    os.makedirs(os.path.join(ROOT, work.replace("../../", "")), exist_ok=True)
    print(f"{name}: {changed} cell(s) patched")
