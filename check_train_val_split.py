#!/usr/bin/env python3
"""Check train/val split: property overlap, category balance, randomization."""
import numpy as np
import pandas as pd
from pathlib import Path

_CSV_PATH = Path(__file__).resolve().parent / "final_clean_dataset.csv"
VAL_RATIO = 0.1
SEED = 42

df = pd.read_csv(_CSV_PATH)
df = df[df["use_in_train"] == True].copy()

# Extract property_id (prefix before _g in image_id)
df["property_id"] = df["image_id"].str.extract(r"^([a-f0-9-]+)_g\d+", expand=False)

n_val = int(len(df) * VAL_RATIO)
n_train = len(df) - n_val
np.random.seed(SEED)
indices = np.random.permutation(len(df))
train_idx = indices[:n_train]
val_idx = indices[n_train:]

train_df = df.iloc[train_idx]
val_df = df.iloc[val_idx]

train_props = set(train_df["property_id"].dropna())
val_props = set(val_df["property_id"].dropna())
overlap = train_props & val_props
train_only = train_props - val_props
val_only = val_props - train_props

print("=" * 60)
print("TRAIN/VAL SPLIT ANALYSIS")
print("=" * 60)
print(f"Train: {n_train} images, {len(train_props)} unique properties")
print(f"Val:   {n_val} images, {len(val_props)} unique properties")
print()
print("Property overlap (data leakage check):")
print(f"  Properties in BOTH train and val: {len(overlap)}")
print(f"  Properties only in train:         {len(train_only)}")
print(f"  Properties only in val:           {len(val_only)}")
if overlap:
    imgs_in_overlap_train = train_df[train_df["property_id"].isin(overlap)].shape[0]
    imgs_in_overlap_val = val_df[val_df["property_id"].isin(overlap)].shape[0]
    print(f"  Images from overlapping props - train: {imgs_in_overlap_train}, val: {imgs_in_overlap_val}")
print()
print("Category distribution (mild/normal/heavy):")
for split_name, sdf in [("Train", train_df), ("Val", val_df)]:
    cats = sdf["category"].value_counts()
    print(f"  {split_name}: {dict(cats)}")
print()
print("Weight distribution:")
for split_name, sdf in [("Train", train_df), ("Val", val_df)]:
    w = sdf["weight"]
    print(f"  {split_name}: 0.3={((w-0.3).abs()<0.01).sum()}, 1.0={((w-1.0).abs()<0.01).sum()}, 5.0={((w-5.0).abs()<0.01).sum()}")
print()
print("Portrait vs landscape:")
for split_name, sdf in [("Train", train_df), ("Val", val_df)]:
    p = sdf["is_portrait"].sum()
    l = len(sdf) - p
    print(f"  {split_name}: portrait={p}, landscape={l}")
print()
print("Conclusion:")
if len(val_only) == 0 and len(overlap) > 0:
    print("  WARNING: All val properties also appear in train (full overlap).")
    print("  Val metrics may be optimistic - model sees same properties in train.")
elif len(val_only) > 0 and len(overlap) > 0:
    print("  Mixed: some val properties are unseen, some overlap with train.")
elif len(val_only) == len(val_props):
    print("  OK: All val properties are unseen (no overlap).")
else:
    print("  Check property_id extraction if counts look wrong.")
