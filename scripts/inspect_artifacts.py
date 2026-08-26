"""Print the schema and shapes of everything under data/. Run from the repo root."""
import os
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


section("aligned_dataset.parquet")
df = pd.read_parquet(os.path.join(DATA, "aligned_dataset.parquet"))
print("shape:", df.shape)
summary = pd.DataFrame({"dtype": df.dtypes.astype(str), "nulls": df.isna().sum()})
print(summary.to_string())
print("\nis_node=True (graph nodes):", int(df["is_node"].sum()))
print("distinct grid zones:", df["grid_zone_id"].nunique(),
      "| distinct countries:", df["country_iso3"].nunique())

for label in ["colocation", "random_control"]:
    section(f"gat_weights_{label}.pt")
    W = torch.load(os.path.join(DATA, "weights", f"gat_weights_{label}.pt"),
                   map_location="cpu", weights_only=False)
    for k, v in W.items():
        if torch.is_tensor(v):
            print(f"  {k:26s} tensor shape={tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"  {k:26s} {type(v).__name__} len={len(v)} = {list(v)}")
        else:
            print(f"  {k:26s} {type(v).__name__} = {v}")

    ei = W["attention_edge_index"]
    n = W["site_id"].numel()
    self_loops = int((ei[0] == ei[1]).sum())
    print(f"\n  nodes={n} edges={ei.shape[1]} self-loops={self_loops} "
          f"avg degree={ei.shape[1] / n:.2f}")

    # site_id in the payload must line up with is_node in the aligned table
    node_ids = set(df.loc[df["is_node"], "site_id"].astype(int))
    payload_ids = set(W["site_id"].tolist())
    print(f"  site_id matches is_node set: {node_ids == payload_ids} "
          f"(payload {len(payload_ids)}, aligned {len(node_ids)})")
