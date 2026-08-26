"""Patch the Kaggle-only notebook so it also runs locally.

Idempotent: re-running makes no further changes. The notebook still works
unmodified on Kaggle -- every patch auto-detects the environment.
"""
import json, os, sys

NB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  "notebooks", "gat_weighted_forecasting.ipynb")

# (old, new, marker) -- `marker` is text unique to the PATCHED form. Detecting
# "already applied" via the new text's first line is wrong: several `old`
# patterns are substrings of their own replacement, which re-patches forever.
PATCHES = [
    # 1. WORK dir: /kaggle/working -> local outputs/ when not on Kaggle
    ('WORK = "/kaggle/working"; os.makedirs(WORK, exist_ok=True)',
     'WORK = "/kaggle/working" if os.path.isdir("/kaggle") else os.path.abspath("../outputs")\n'
     'os.makedirs(WORK, exist_ok=True)',
     'os.path.isdir("/kaggle")'),

    # 2. INPUT_ROOT: /kaggle/input -> local data/ when not on Kaggle
    ('INPUT_ROOT = "/kaggle/input"',
     'INPUT_ROOT = "/kaggle/input" if os.path.isdir("/kaggle/input") else os.path.abspath("../data")',
     'os.path.abspath("../data")'),

    # 3. torch_geometric is imported but never used -- drop the heavy dep
    ('import torch_geometric',
     '# import torch_geometric  # unused: the .pt payloads are plain dicts, torch.load is enough',
     '# import torch_geometric'),

    # 4. never hardcode a key
    ('OPENAI_API_KEY = "sk-REPLACE-WITH-YOUR-OPENAI-KEY"',
     'OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")',
     'os.environ.get("OPENAI_API_KEY"'),
]

nb = json.load(open(NB, encoding="utf-8"))
applied, already = [], []

for old, new, marker in PATCHES:
    hit = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if marker in src:
            already.append(old[:48]); hit = True; break
        if old in src:
            cell["source"] = (src.replace(old, new)).splitlines(keepends=True)
            applied.append(old[:48]); hit = True; break
    if not hit:
        print(f"  !! pattern not found: {old[:60]!r}")

if applied:
    json.dump(nb, open(NB, "w", encoding="utf-8"), indent=1, ensure_ascii=False)

print(f"patched : {len(applied)}")
for a in applied: print("   +", a)
print(f"already : {len(already)}")
for a in already: print("   =", a)
