#!/usr/bin/env python
"""
Check that every WSI in a slide list has its <slide_id>.npy embedding at
    <folder>/<project>/<encoder>/<slide_id>.npy
(the path MIL-Adapter's utils.load_data builds).

Examples
--------
python check_npy.py --folder /project/rrg-josedolz/natgill/data \
    --project NSCLC --encoder CONCH --csv /project/rrg-josedolz/natgill/baselines/MIL-Adapter/local_data/csv/NSCLC.csv 

# check several encoders at once, also try np.load on each file
python check_npy.py --folder ... --project NSCLC --encoder CONCH UNI \
    --csv labels.csv --col slide_id --try-load
"""
import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd


def read_ids(args):
    if args.csv:
        df = pd.read_csv(args.csv)
        if args.col not in df.columns:
            sys.exit(f"Column '{args.col}' not in {args.csv}. Columns: {list(df.columns)}")
        ids = df[args.col].dropna().astype(str).tolist()
    else:
        with open(args.txt) as f:
            ids = [l.strip() for l in f if l.strip()]
    # strip extensions if the list contains full file names
    ids = [os.path.splitext(i)[0] if i.endswith((".svs", ".tif", ".tiff", ".ndpi", ".npy")) else i
           for i in ids]
    dup = len(ids) - len(set(ids))
    if dup:
        print(f"[warn] {dup} duplicate slide IDs in the list")
    return list(dict.fromkeys(ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--encoder", nargs="+", required=True)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="CSV holding the slide IDs")
    src.add_argument("--txt", help="text file, one slide ID per line")
    p.add_argument("--col", default="WSI", help="column name in --csv")
    p.add_argument("--try-load", action="store_true", help="also np.load each file to catch corrupted ones")
    p.add_argument("--out", default="missing_npy.txt")
    args = p.parse_args()

    ids = read_ids(args)
    print(f"{len(ids)} slides in list")
    all_missing = []

    for enc in args.encoder:
        d = os.path.join(args.folder, args.project, enc)
        print(f"\n=== {enc}: {d}")
        if not os.path.isdir(d):
            print("  [error] directory does not exist")
            all_missing += [(enc, i, "no dir") for i in ids]
            continue

        files = os.listdir(d)
        present = {os.path.splitext(f)[0] for f in files if f.endswith(".npy")}
        other_ext = defaultdict(list)
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext != ".npy":
                other_ext[stem].append(ext)
        # index by TCGA slide barcode (first 23 chars, e.g. TCGA-77-A5GF-01Z-00-DX1)
        by_barcode = defaultdict(list)
        for s in present:
            by_barcode[s[:23]].append(s)

        missing = [i for i in ids if i not in present]
        print(f"  found {len(ids) - len(missing)}/{len(ids)}  |  missing {len(missing)}")
        print(f"  .npy files in dir not in list: {len(present - set(ids))}")

        for i in missing:
            hint = ""
            if i in other_ext:
                hint = f"exists with other extension {other_ext[i]}"
            elif by_barcode.get(i[:23]):
                hint = f"same barcode, different name: {by_barcode[i[:23]][:2]}"
            print(f"  MISSING {i}" + (f"   <- {hint}" if hint else ""))
            all_missing.append((enc, i, hint))

        if args.try_load:
            bad = []
            for i in ids:
                if i in present:
                    try:
                        a = np.load(os.path.join(d, f"{i}.npy"), mmap_mode="r")
                        if a.size == 0:
                            bad.append((i, "empty"))
                    except Exception as e:
                        bad.append((i, repr(e)))
            print(f"  unreadable/empty: {len(bad)}")
            for i, e in bad:
                print(f"  BAD {i}: {e}")
                all_missing.append((enc, i, f"bad file: {e}"))

    with open(args.out, "w") as f:
        for enc, i, h in all_missing:
            f.write(f"{enc}\t{i}\t{h}\n")
    print(f"\nWrote {len(all_missing)} problem entries to {args.out}")
    sys.exit(1 if all_missing else 0)


if __name__ == "__main__":
    main()