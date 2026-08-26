"""Check every input the notebook REQUIRES before you spend time running it.

Mirrors the resolution logic in notebook Section 1, but reports the full
picture instead of raising on the first miss.
"""
import os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

def find(pred, root=DATA):
    for r, dirs, files in os.walk(root):
        for f in files:
            if pred(f):
                return os.path.join(r, f)
    return None

CHECKS = {
    "aligned":  (lambda f: f == "aligned_dataset.parquet",                      "have it"),
    "w_coloc":  (lambda f: f == "gat_weights_colocation.pt",                    "have it"),
    "w_random": (lambda f: f == "gat_weights_random_control.pt",                "have it"),
    "ember_us": (lambda f: f.endswith(".csv") and "us_monthly" in f.lower(),    "Ember monthly, US"),
    "ember_eu": (lambda f: f.endswith(".csv") and "europe_monthly" in f.lower(),"Ember monthly, Europe"),
    "ember_in": (lambda f: f.endswith(".csv") and "india_monthly" in f.lower(), "Ember monthly, India"),
    "g3p_tws":  (lambda f: f.endswith(".csv") and "tws_rivbas" in f.lower(),    "G3P basin water storage"),
}

print(f"scanning {DATA}\n")
missing = []
for key, (pred, note) in CHECKS.items():
    p = find(pred)
    if p:
        size = os.path.getsize(p) / 1e6
        print(f"  [ok]   {key:9s} {os.path.relpath(p, ROOT):<52} {size:7.2f} MB")
    else:
        print(f"  [MISS] {key:9s} {'-':<52} {note}")
        missing.append(key)

print()
if missing:
    print(f"{len(missing)} required input(s) missing: {', '.join(missing)}")
    print("Section 1 will raise FileNotFoundError. Drop the CSVs anywhere under data/")
    print("-- filename matching is by substring, so the vendor's own names work as-is.")
    sys.exit(1)
print("All required inputs present. The notebook will run end to end.")

# optional deps
for m, why in [("openai", "Section 6B (Nexus) only; SARIMA runs without it")]:
    try:
        __import__(m)
    except ImportError:
        print(f"note: '{m}' not installed -- {why}")
