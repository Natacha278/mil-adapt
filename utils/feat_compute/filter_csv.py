#!/usr/bin/env python
"""
Create a CSV keeping only the slides whose embedding exists at
    {folder}/{project}/{encoder}/{WSI}.npy
(same path as MIL-Adapter's utils.load_data). Run from the MIL-Adapter repo root.

    python filter_csv.py --folder /project/rrg-josedolz/natgill/data --project NSCLC --encoder CONCH
        -> writes ./local_data/csv/NSCLC_available.csv

    python filter_csv.py ... --replace
        -> backs up NSCLC.csv to NSCLC_full.csv, then writes the filtered rows to NSCLC.csv
           (the file load_data reads)
"""
import argparse
import os
import shutil

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--folder", required=True)
p.add_argument("--project", default="NSCLC")
p.add_argument("--encoder", default="CONCH")
p.add_argument("--csv", default=None, help="input CSV (default ./local_data/csv/{project}.csv)")
p.add_argument("--out", default=None, help="output CSV (default ./local_data/csv/{project}_available.csv)")
p.add_argument("--replace", action="store_true", help="overwrite {project}.csv, keeping a *_full.csv backup")
args = p.parse_args()

csv_in = args.csv or f"./local_data/csv/{args.project}.csv"
emb_dir = os.path.join(args.folder, args.project, args.encoder)

data = pd.read_csv(csv_in)
present = {f[:-4] for f in os.listdir(emb_dir) if f.endswith(".npy")}
keep = data["WSI"].astype(str).isin(present)

print(f"Input : {csv_in} ({len(data)} rows)")
print(f"Embeddings dir: {emb_dir} ({len(present)} .npy files)")
print(f"Kept {keep.sum()}  |  dropped {(~keep).sum()}")
for w in data.loc[~keep, "WSI"]:
    print(f"  dropped: {w}")
if "GT" in data.columns:
    print(f"Classes before: {data['GT'].value_counts().to_dict()}")
    print(f"Classes after : {data.loc[keep, 'GT'].value_counts().to_dict()}")

if args.replace:
    backup = csv_in.replace(".csv", "_full.csv")
    if not os.path.exists(backup):  # never overwrite an existing backup
        shutil.copy(csv_in, backup)
        print(f"Backup: {backup}")
    out = csv_in
else:
    out = args.out or os.path.join(os.path.dirname(csv_in), f"{args.project}_available.csv")

data[keep].to_csv(out, index=False)
print(f"Wrote {out}")