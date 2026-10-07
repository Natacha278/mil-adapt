"""
add_tumor_pct_to_csv.py

Adds a tumor_patch_pct column (and n_patches, for context) to the CAMELYON16 CSV
(local_data/csv/CAMELYON16.csv, columns: case_id, WSI, GT), computed per slide.

Reuses compute_tumor_patch_pct from analyze_tumor_pct_accuracy.py (which in turn reuses
parse_camelyon16_xml from visu_ABMIL.py), so the definition is identical to what the
accuracy-vs-tumor-pct analysis already uses -- this script does NOT redefine it.

IMPORTANT: tumor_patch_pct is computed relative to the number of patches TRIDENT
actually extracted for that slide (the .h5 file's "coords" array), i.e. AFTER TRIDENT's
own tissue-detection/background filtering -- not a naive full grid over the whole slide
image. So this is "% of the tissue TRIDENT kept that is tumor" (= % of what the MIL
model actually sees), not "% of the raw slide image that is tumor". Normal slides
(no XML annotation) get tumor_patch_pct = 0.0.

Run on a login node is fine here -- this only reads h5 coords/attrs and XML files,
no model loading and no GPU needed.
"""

import os
import argparse
import shutil

import pandas as pd
from tqdm import tqdm

from analyze_tumor_pct_accuracy import compute_tumor_patch_pct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="./local_data/csv/CAMELYON16.csv",
                         help="Existing CAMELYON16 CSV (columns: case_id, WSI, GT) to augment.")
    parser.add_argument("--h5_dir",
                         default="/project/rrg-josedolz/natgill/data/camelyon16/trident/20x_512px_0px_overlap/features_conch_v1",
                         help="Original TRIDENT h5 files (coords + patch_size_level0).")
    parser.add_argument("--xml_dir", required=True,
                         help="Directory containing lesion_annotations/*.xml.")
    parser.add_argument("--out_csv", default=None,
                         help="Where to write the augmented CSV. Defaults to overwriting "
                              "--csv_path in place (a .bak backup of the original is made "
                              "first). Pass a different path to write a separate file instead.")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")

    df = pd.read_csv(args.csv_path)
    for required_col in ("WSI", "GT"):
        if required_col not in df.columns:
            raise ValueError(f"Expected column '{required_col}' in {args.csv_path}, "
                              f"found columns: {list(df.columns)}")

    tumor_pcts, n_patches_list = [], []
    for slide_id in tqdm(df["WSI"].values, desc="Computing per-slide tumor-patch %"):
        pct, n_patches = compute_tumor_patch_pct(slide_id, args.h5_dir, args.xml_dir)
        tumor_pcts.append(pct)
        n_patches_list.append(n_patches)

    df["tumor_patch_pct"] = tumor_pcts
    df["n_patches"] = n_patches_list

    # Sanity check: every Normal slide should show 0% (no XML). Flag anything unexpected
    # rather than silently writing a CSV that looks wrong later.
    normal_nonzero = df[(df["GT"] == "Normal") & (df["tumor_patch_pct"] > 0)]
    if len(normal_nonzero) > 0:
        print(f"WARNING: {len(normal_nonzero)} slide(s) labeled 'Normal' have tumor_patch_pct > 0 "
              f"-- check these for a mislabeled GT or a stray/mismatched XML file:")
        print(normal_nonzero[["WSI", "GT", "tumor_patch_pct"]].to_string(index=False))

    tumor_zero = df[(df["GT"] == "Tumor") & (df["tumor_patch_pct"] == 0)]
    if len(tumor_zero) > 0:
        print(f"\nNOTE: {len(tumor_zero)} slide(s) labeled 'Tumor' have tumor_patch_pct == 0 "
              f"-- the annotated lesion may be too small to register at any patch's center, "
              f"or the XML file may be missing for these slides. Worth spot-checking:")
        print(tumor_zero[["WSI", "GT", "tumor_patch_pct"]].to_string(index=False))

    out_csv = args.out_csv or args.csv_path
    if out_csv == args.csv_path:
        backup_path = args.csv_path + ".bak"
        shutil.copy2(args.csv_path, backup_path)
        print(f"\nBacked up original CSV to: {backup_path}")

    df.to_csv(out_csv, index=False)
    print(f"Saved augmented CSV ({len(df)} slides) to: {out_csv}")
    print(f"\nSummary:")
    print(f"  Normal slides: {(df['GT'] == 'Normal').sum()}, all at tumor_patch_pct=0 "
          f"unless flagged above")
    tumor_df = df[df["GT"] == "Tumor"]
    if len(tumor_df) > 0:
        print(f"  Tumor slides: {len(tumor_df)}, tumor_patch_pct range "
              f"[{tumor_df['tumor_patch_pct'].min():.2f}, {tumor_df['tumor_patch_pct'].max():.2f}], "
              f"median {tumor_df['tumor_patch_pct'].median():.2f}")


if __name__ == "__main__":
    main()
