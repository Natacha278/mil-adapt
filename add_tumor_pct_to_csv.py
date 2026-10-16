"""
add_tumor_pct_to_csv.py

Adds per-slide witness-rate columns to local_data/csv/CAMELYON16.csv
(columns case_id, WSI, GT), computed by utils/witness_rate.py.

WHY THIS WAS REWRITTEN
----------------------
The previous version imported compute_tumor_patch_pct from analyze_tumor_pct_accuracy.py,
which no longer defines it, and that function read the patch size from the h5 file ROOT
attributes, falling back to 512 level-0 px while CAMELYON16 patches at 20x / 512 px span
1024. Every patch was then tested at the centre of its top-left quarter, so the old
tumor_patch_pct is shifted and small lesions can read 0%. It also counted the "_2"
groups (non-tumor holes inside a tumor region) as tumor.

COLUMNS WRITTEN (all percentages are of the slide's TISSUE patches, i.e. what TRIDENT
extracted and the MIL model sees)
    tumor_patch_pct     patches whose centre is in tumor -- the historical definition,
                        corrected. Kept under this name so every existing analysis
                        (analyze_tumor_ratio_*.py, utils/tumor_ratio_metrics.py) uses
                        the corrected values without any change.
    tumor_area_pct      tumor area / tissue area (sum of exact patch occupancies / N).
                        Does not depend on where the patch grid falls; preferred x-axis.
    touched_pct         patches with any tumor in them.
    witness_occ_pct     patches at least --min_occupancy covered.
    n_patches, n_tumor_patches, n_touched_patches, n_lesions, patch_size_level0,
    max_lesion_diam_mm, lesion_stage (largest lesion: ITC / micro / macro)
The previous tumor_patch_pct is kept as tumor_patch_pct_old for comparison, and a .bak
copy of the input CSV is written before overwriting it.

Annotation source per tumor slide: <xml_dir>/<WSI>.xml (exact polygons). With
--mask_dir, <mask_dir>/<WSI>_mask.tif is used instead when present (raster at
--mask_level). Normal slides need neither.

Login node is fine: reads h5 coords/attrs and XML only. --mask_dir needs openslide.

    python add_tumor_pct_to_csv.py --xml_dir $SCRATCH/camelyon16/annotations
"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.witness_rate import compute_slide, DEFAULT_MPP_LEVEL0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", default="./local_data/csv/CAMELYON16.csv")
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--xml_dir", required=True, help="Directory with <WSI>.xml annotations.")
    p.add_argument("--mask_dir", default=None,
                   help="Optional directory with <WSI>_mask.tif; used instead of the XML "
                        "when present.")
    p.add_argument("--mask_level", type=int, default=4)
    p.add_argument("--min_occupancy", type=float, default=0.5)
    p.add_argument("--mpp_level0", type=float, default=DEFAULT_MPP_LEVEL0)
    p.add_argument("--lesion_csv", default=None,
                   help="Also write a per-lesion table (area, diameters, stage, patches).")
    p.add_argument("--out_csv", default=None,
                   help="Defaults to overwriting --csv_path (after a .bak backup).")
    args = p.parse_args()

    df = pd.read_csv(args.csv_path)
    for col in ("WSI", "GT"):
        if col not in df:
            raise SystemExit(f"{args.csv_path} has no '{col}' column")

    rows, lesion_tables, missing_h5, missing_ann = [], [], [], []
    for wsi, gt in tqdm(list(zip(df["WSI"], df["GT"])), desc="witness rate"):
        h5 = os.path.join(args.h5_dir, f"{wsi}.h5")
        if not os.path.exists(h5):
            missing_h5.append(wsi)
            rows.append({})
            continue
        xml = os.path.join(args.xml_dir, f"{wsi}.xml")
        mask = (os.path.join(args.mask_dir, f"{wsi}_mask.tif") if args.mask_dir else None)
        kw = {}
        if gt == "Tumor":
            if mask and os.path.exists(mask):
                kw = {"mask_path": mask, "mask_level": args.mask_level}
            elif os.path.exists(xml):
                kw = {"xml_path": xml}
            else:
                missing_ann.append(wsi)
        elif os.path.exists(xml):
            print(f"NOTE: {wsi} is labelled Normal but has an annotation file; ignored.")

        stats, _patches, lesions = compute_slide(h5, min_occupancy=args.min_occupancy,
                                                 mpp_level0=args.mpp_level0,
                                                 with_lesions=True, **kw)
        row = {"n_patches": stats["n_patches"],
               "tumor_patch_pct": stats["witness_rate_centre_pct"],
               "tumor_area_pct": stats["tumor_area_pct"],
               "touched_pct": stats["touched_pct"],
               "witness_occ_pct": stats["witness_rate_occ_pct"],
               "n_tumor_patches": stats["n_tumor_centre"],
               "n_touched_patches": stats["n_touched"],
               "n_lesions": stats["n_lesions"],
               "patch_size_level0": stats["patch_size_level0"],
               "annotation_source": stats["annotation_source"],
               "max_lesion_diam_mm": np.nan, "lesion_stage": ""}
        if lesions is not None and len(lesions):
            big = lesions.loc[lesions["diam_max_mm"].idxmax()]
            row["max_lesion_diam_mm"] = big["diam_max_mm"]
            row["lesion_stage"] = big["stage"]
            lesions.insert(0, "WSI", wsi)
            lesion_tables.append(lesions)
        rows.append(row)

    new = pd.DataFrame(rows, index=df.index)
    if "tumor_patch_pct" in df:
        df = df.rename(columns={"tumor_patch_pct": "tumor_patch_pct_old"})
    for col in new.columns:
        if col in df and col != "tumor_patch_pct":
            df = df.drop(columns=col)
    df = pd.concat([df, new], axis=1)

    # sanity checks: say what looks wrong instead of writing it silently
    ps = df["patch_size_level0"].dropna().unique()
    print(f"\npatch_size_level0 found: {sorted(int(v) for v in ps)}")
    bad_norm = df[(df.GT == "Normal") & (df.tumor_area_pct > 0)]
    if len(bad_norm):
        print(f"WARNING: {len(bad_norm)} Normal slide(s) with tumor area > 0:\n"
              f"{bad_norm[['WSI', 'tumor_area_pct']].to_string(index=False)}")
    tum = df[df.GT == "Tumor"]
    zero_c = tum[(tum.tumor_patch_pct == 0) & (tum.tumor_area_pct > 0)]
    zero_a = tum[tum.tumor_area_pct == 0]
    print(f"Tumor slides: {len(tum)}; centre rule 0% but tumor present in some patch: "
          f"{len(zero_c)} (lesion straddles patch boundaries or is very small); "
          f"no tumor in any extracted patch: {len(zero_a)}")
    if len(zero_a):
        print(zero_a[["WSI", "n_lesions", "max_lesion_diam_mm"]].to_string(index=False))
    if missing_h5:
        print(f"WARNING: {len(missing_h5)} slide(s) without h5 (columns left empty): "
              f"{missing_h5[:10]}")
    if missing_ann:
        print(f"WARNING: {len(missing_ann)} Tumor slide(s) without XML/mask (0% written): "
              f"{missing_ann[:10]}")
    if "tumor_patch_pct_old" in df:
        t = df[df.GT == "Tumor"].dropna(subset=["tumor_patch_pct", "tumor_patch_pct_old"])
        moved = (t.tumor_patch_pct - t.tumor_patch_pct_old).abs()
        print(f"vs old tumor_patch_pct (tumor slides): median |change| {moved.median():.3f} "
              f"points, max {moved.max():.3f}; old 0% -> now > 0%: "
              f"{int(((t.tumor_patch_pct_old == 0) & (t.tumor_patch_pct > 0)).sum())}")

    out = args.out_csv or args.csv_path
    if out == args.csv_path:
        shutil.copy2(args.csv_path, args.csv_path + ".bak")
        print(f"backup -> {args.csv_path}.bak")
    df.to_csv(out, index=False)
    print(f"saved {len(df)} slides -> {out}")
    if args.lesion_csv and lesion_tables:
        pd.concat(lesion_tables, ignore_index=True).to_csv(args.lesion_csv, index=False)
        print(f"per-lesion table -> {args.lesion_csv}")


if __name__ == "__main__":
    main()
