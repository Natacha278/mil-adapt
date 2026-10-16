"""
analyze_zeroshot_dice.py

Dice of CONCH zero-shot PATCH predictions against the CAMELYON16 tumor annotations,
per tumor slide, overall and as a function of witness rate.

PREDICTION (per patch)
    P(Tumor) = softmax(logit_scale * [cos(p, t_Normal), cos(p, t_Tumor)])[Tumor]
    predicted tumor  <=>  P(Tumor) > --threshold   (0.5 = argmax, as before)

GROUND TRUTH (per patch), from utils/witness_rate.py: the true patch footprint (1024
level-0 px at 20x/512 px on CAMELYON16, read from the h5 coords attributes) and the
"_2" non-tumor holes subtracted. Three definitions, all reported:
    dice_centre   GT = patch centre inside tumor. Same rule as the previous version of
                  this script (whose 512-px footprint tested the centre of each patch's
                  top-left quarter); column `dice` keeps this value for continuity.
    dice_occ      GT = at least --min_occupancy of the patch is tumor (default 0.5).
    dice_area     area-weighted ("soft") Dice:  2 * sum(pred_i * occ_i)
                  / (sum(pred_i) + sum(occ_i)),  occ_i = exact tumor fraction of patch i.
                  Does not depend on where the patch grid falls relative to the lesion,
                  and does not collapse to 0 or undefined for lesions smaller than a
                  patch. Recommended headline.
A slide where both the prediction and the GT are empty has an UNDEFINED Dice (NaN,
counted and excluded from means) rather than the previous convention of 1.0.

REPORTED
  * per tumor slide: counts (tp/fp/fn/tn for the centre rule), the three Dice values,
    witness rate (tumor area %, centre-rule %), largest lesion diameter and stage;
  * mean / median per-slide Dice, and pooled (micro) Dice over all patches;
  * the same by tumor-area bin and by lesion stage (ITC / micro / macro);
  * normal slides: fraction of patches and patches per slide predicted tumor;
  * a figure of per-slide Dice vs tumor area %.

SCORES. By default CONCH scores the patches from the TRIDENT h5 features (GPU). With
--score_table, the margins sim(Tumor) - sim(Normal) are read from a per-patch table
(analyze_zeroshot_score_vs_lesion_size.py --save_patches; columns WSI, x, y, score) and
P(Tumor) = sigmoid(logit_scale * margin); no CONCH needed. Its labels are NOT used (they
carry the old patch-size bug); GT is always recomputed here. A --probe table keeps only
300 normal patches per slide, so normal-slide false positives would be undercounted.

USAGE
    python analyze_zeroshot_dice.py --xml_dir $SCRATCH/camelyon16/annotations \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --split all --out_csv res_classif/zs_dice_test_descr.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

CLASSES = ["Normal", "Tumor"]
CONCH_LOGIT_SCALE = 56.347694396972656        # exp(logit_scale) of CONCH, as in utils/adapters.py
AREA_BINS = [0.0, 0.1, 0.5, 2.0, 10.0, 100.0]  # tumor area, % of tissue
STAGES = ["ITC", "micro", "macro"]


# =====================================================================================
# Dice
# =====================================================================================

def dice_binary(pred, gt):
    """2TP / (2TP + FP + FN); NaN when both masks are empty (undefined)."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp = int((pred & gt).sum())
    den = int(pred.sum()) + int(gt.sum())
    return np.nan if den == 0 else 2.0 * tp / den


def dice_area(pred, occ):
    """Area-weighted Dice: overlap counted in tumor AREA rather than in patches."""
    pred, occ = np.asarray(pred, float), np.asarray(occ, float)
    den = pred.sum() + occ.sum()
    return np.nan if den == 0 else 2.0 * float((pred * occ).sum()) / den


def slide_record(wsi, p_tumor, occ, centre, threshold, min_occupancy):
    pred = p_tumor > threshold
    gt_c = np.asarray(centre, bool)
    gt_o = np.asarray(occ) >= min_occupancy
    tp = int((pred & gt_c).sum()); fp = int((pred & ~gt_c).sum())
    fn = int((~pred & gt_c).sum()); tn = int((~pred & ~gt_c).sum())
    d_c = dice_binary(pred, gt_c)
    return {
        "WSI": wsi, "n_patches": len(pred),
        "n_gt_tumor_patches": int(gt_c.sum()), "n_pred_tumor_patches": int(pred.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "dice": d_c, "dice_centre": d_c,
        "dice_occ": dice_binary(pred, gt_o), "n_gt_occ_patches": int(gt_o.sum()),
        "dice_area": dice_area(pred, occ),
        # pooled-Dice accumulators
        "_area_tp": float((pred * occ).sum()), "_area_gt": float(np.sum(occ)),
        "_occ_tp": int((pred & gt_o).sum()),
    }


def pooled(df):
    """Micro Dice over all patches of all slides in df."""
    out = {}
    den = 2 * df.tp.sum() + df.fp.sum() + df.fn.sum()
    out["dice_centre"] = 2 * df.tp.sum() / den if den else np.nan
    den = df.n_pred_tumor_patches.sum() + df.n_gt_occ_patches.sum()
    out["dice_occ"] = 2 * df._occ_tp.sum() / den if den else np.nan
    den = df.n_pred_tumor_patches.sum() + df._area_gt.sum()
    out["dice_area"] = 2 * df._area_tp.sum() / den if den else np.nan
    return out


def summarise(df, by):
    rows = []
    for key, g in df.groupby(by, observed=False, sort=False):
        r = {by: key, "n_slides": len(g)}
        for m in ["dice_area", "dice_centre", "dice_occ"]:
            r[f"{m}_mean"] = g[m].mean()
            r[f"{m}_median"] = g[m].median()
            r[f"{m}_n_undefined"] = int(g[m].isna().sum())
        r.update({f"{k}_pooled": v for k, v in pooled(g).items()})
        rows.append(r)
    return pd.DataFrame(rows)


# =====================================================================================
# Figure
# =====================================================================================

def make_figure(df, normal_fp, headline, threshold, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink2, grid = "#52514e", "#e4e3df"
    colors = {"ITC": "#e34948", "micro": "#eda100", "macro": "#2a78d6"}
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for st in STAGES:
        g = df[(df.lesion_stage == st) & df[headline].notna()]
        ax.scatter(g.tumor_area_pct.clip(lower=1e-3), g[headline], s=26, color=colors[st],
                   edgecolor="white", linewidth=0.6, label=f"{st} (n={len(g)})", zorder=3)
    b = df.dropna(subset=[headline]).copy()
    b["bin"] = pd.cut(b.tumor_area_pct, AREA_BINS, include_lowest=True)
    med = b.groupby("bin", observed=True).agg(x=("tumor_area_pct", "median"),
                                             y=(headline, "median"))
    ax.plot(med.x, med.y, "D-", color="#52514e", lw=1.4, ms=7, mfc="none", mew=1.2,
            label="bin median", zorder=2)          # under the points, hollow: never hides one
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("tumor area, % of tissue")
    ax.set_ylabel({"dice_area": "area-weighted Dice", "dice_centre": "Dice (centre-rule GT)",
                   "dice_occ": "Dice (occupancy GT)"}[headline])
    ax.grid(color=grid, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Zero-shot patch segmentation, per tumor slide", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    n_undef = int(df[headline].isna().sum())
    fig.text(0.01, 0.005,
             f"P(Tumor) > {threshold:g}.  Normal slides: {normal_fp:.2%} of patches predicted tumor."
             + (f"  {n_undef} slide(s) with undefined Dice not shown." if n_undef else "")
             + "  Tumor area below 0.001% drawn at 0.001%.",
             fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================================================
# Main
# =====================================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--xml_dir", required=True)
    p.add_argument("--csv_path", default=None,
                   help="Slide list + GT (default ./local_data/csv/<project>.csv).")
    p.add_argument("--folder", default=None, help="Unused; kept for old command lines.")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--checkpoint_path",
                   default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--score_table", default=None,
                   help="Read per-patch margins from this table instead of running CONCH.")
    p.add_argument("--logit_scale", type=float, default=CONCH_LOGIT_SCALE,
                   help="Only with --score_table (CONCH's own value is used otherwise).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Patch is predicted tumor if P(Tumor) > threshold. 0.5 = argmax.")
    p.add_argument("--min_occupancy", type=float, default=0.5)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--headline", choices=["dice_area", "dice_centre", "dice_occ"],
                   default="dice_area", help="Metric plotted and printed first.")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--split", choices=["all", "test", "train"], default="all",
                   help="CAMELYON16 official split by name prefix (see --test_prefix).")
    p.add_argument("--test_prefix", default="test_")
    p.add_argument("--out_csv", default="zeroshot_dice_per_slide.csv")
    args = p.parse_args()

    from tqdm import tqdm
    from utils import witness_rate as W

    meta = pd.read_csv(args.csv_path or f"./local_data/csv/{args.project}.csv")
    if args.split != "all":
        is_test = meta.WSI.astype(str).str.startswith(args.test_prefix)
        meta = meta[is_test if args.split == "test" else ~is_test]
        print(f"--split {args.split}: {len(meta)} slides")
    mpp = args.mpp_level0 or W.DEFAULT_MPP_LEVEL0

    from utils import zeroshot as Z
    scorer = Z.make_scorer(args.score_table, args.checkpoint_path, args.text_prototypes,
                           args.logit_scale, args.batch_size)

    rows, normal_rows = [], []
    missing = {"h5": [], "xml": [], "scores": []}
    for wsi, gt in tqdm(list(zip(meta.WSI, meta.GT)), desc="zero-shot Dice"):
        h5 = os.path.join(args.h5_dir, f"{wsi}.h5")
        if not os.path.exists(h5):
            missing["h5"].append(wsi)
            continue
        xml = os.path.join(args.xml_dir, f"{wsi}.xml")
        if gt == "Tumor" and not os.path.exists(xml):
            missing["xml"].append(wsi)
            continue
        grid = W.load_patch_grid(h5, with_features=scorer.needs_features)
        margin = scorer(wsi, grid)
        if margin is None:
            missing["scores"].append(wsi)
            continue
        p_tumor = Z.p_tumor_from_margin(margin, scorer.scale)
        if gt == "Normal":
            normal_rows.append({"WSI": wsi, "n_patches": len(p_tumor),
                                "n_pred_tumor_patches": int((p_tumor > args.threshold).sum())})
            continue
        ann = W.parse_camelyon16_xml(xml)
        occ = W.patch_occupancy_from_polygons(grid.coords, grid.patch_size, ann)
        centre = W.patch_centre_in_polygons(grid.coords, grid.patch_size, ann)
        rec = slide_record(wsi, p_tumor, occ, centre, args.threshold, args.min_occupancy)
        st = W.slide_witness_stats(occ, centre, args.min_occupancy)
        geo = W.lesion_geometry(ann, mpp)
        rec.update({"tumor_area_pct": st["tumor_area_pct"],
                    "witness_rate_centre_pct": st["witness_rate_centre_pct"],
                    "n_lesions": len(ann), "patch_size_level0": grid.patch_size,
                    "max_lesion_diam_mm": float(geo.diam_max_mm.max()) if len(geo) else np.nan,
                    "lesion_stage": (geo.loc[geo.diam_max_mm.idxmax(), "stage"]
                                     if len(geo) else "none")})
        rows.append(rec)

    if not rows:
        raise SystemExit("no tumor slide evaluated -- check --h5_dir / --xml_dir / --split")
    df = pd.DataFrame(rows)
    df["area_bin"] = pd.cut(df.tumor_area_pct, AREA_BINS, include_lowest=True).astype(str)
    nd = pd.DataFrame(normal_rows)

    base = os.path.splitext(args.out_csv)[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.drop(columns=[c for c in df if c.startswith("_")]).to_csv(args.out_csv, index=False)
    by_bin = summarise(df, "area_bin")
    by_bin["order"] = by_bin.area_bin.map(
        {str(c): i for i, c in enumerate(pd.cut(pd.Series(AREA_BINS[1:]), AREA_BINS,
                                                 include_lowest=True).cat.categories)})
    by_bin = by_bin.sort_values("order").drop(columns="order")
    by_stage = summarise(df, "lesion_stage")
    by_stage["order"] = by_stage.lesion_stage.map({s: i for i, s in enumerate(STAGES)})
    by_stage = by_stage.sort_values("order").drop(columns="order")
    by_bin.to_csv(f"{base}_by_area_bin.csv", index=False)
    by_stage.to_csv(f"{base}_by_stage.csv", index=False)
    normal_fp = (nd.n_pred_tumor_patches.sum() / nd.n_patches.sum()) if len(nd) else np.nan
    if len(nd):
        nd.to_csv(f"{base}_normal_slides.csv", index=False)
    make_figure(df, normal_fp if np.isfinite(normal_fp) else 0.0, args.headline,
                args.threshold, f"{base}.png")

    # ---------------------------------------------------------------- report
    h = args.headline
    order = [h] + [m for m in ["dice_area", "dice_centre", "dice_occ"] if m != h]
    pool = pooled(df)
    print("\n" + "=" * 92)
    print(f"Zero-shot patch Dice  [{len(df)} tumor slides, P(Tumor) > {args.threshold:g}, "
          f"patch size {sorted(df.patch_size_level0.unique().tolist())} level-0 px, "
          f"prompts {os.path.basename(args.text_prototypes) if not args.score_table else 'from table'}]")
    print("=" * 92)
    for m in order:
        v = df[m]
        print(f"{m:<12} mean {v.mean():.4f}  median {v.median():.4f}  pooled {pool[m]:.4f}"
              f"  (undefined on {int(v.isna().sum())} slide(s))")
    show = ["n_slides"] + [f"{m}_{s}" for m in order[:2] for s in ("mean", "median")]
    print("-" * 92 + "\nby tumor area (% of tissue):")
    print(by_bin[["area_bin"] + show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("-" * 92 + "\nby largest lesion stage:")
    print(by_stage[["lesion_stage"] + show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("-" * 92)
    if len(nd):
        print(f"normal slides: {len(nd)}, {normal_fp:.4%} of patches predicted tumor "
              f"(median {nd.n_pred_tumor_patches.median():.0f} patches per slide)")
    for k, v in missing.items():
        if v:
            print(f"skipped, no {k}: {len(v)} ({v[:5]}{'...' if len(v) > 5 else ''})")
    print("=" * 92)
    print(f"-> {args.out_csv}, {base}_by_area_bin.csv, {base}_by_stage.csv, {base}.png")


if __name__ == "__main__":
    main()
