"""
analyze_patch_accuracy_vs_witness_rate.py

CONCH zero-shot PATCH classification on CAMELYON16 as a function of each slide's
witness rate: accuracy-type metrics at a threshold, and threshold-free AUCs.

PREDICTION   P(Tumor) = softmax(logit_scale * [cos(p, t_Normal), cos(p, t_Tumor)])[Tumor]
             = sigmoid(logit_scale * margin);  predicted tumor <=> P(Tumor) > --threshold
GROUND TRUTH utils/witness_rate.py: true patch footprint (coords attrs), "_2" holes
             subtracted.  --gt centre (default): tumor iff the patch centre is in tumor;
             --gt occ: tumor iff at least --min_occupancy of the patch is tumor.
             --exclude_partial keeps only clean patches (occupancy 0, or >= 0.99) to see
             how much of the small-lesion behaviour is label noise at lesion borders.
WITNESS RATE --wr tumor_area_pct (default: tumor area / tissue area, grid-independent)
             or witness_rate_centre_pct (share of patches whose centre is in tumor).

PER TUMOR SLIDE
  accuracy            correct patches / all evaluated patches of the slide. At low witness
                      rate almost every patch is normal, so this is ~specificity: it does
                      NOT tell whether the lesion was found. Reported because it was asked
                      for; read it next to sensitivity.
  sensitivity         tumor patches predicted tumor
  specificity         normal patches of the slide predicted normal
  balanced_accuracy   (sensitivity + specificity) / 2
  auc_within          AUC of the margin, tumor patches vs the SAME slide's normal patches:
                      does the encoder rank this lesion above its surroundings?
  auc_vs_normal_slides AUC of the margin, tumor patches vs ALL patches of the normal slides
                      evaluated: does the lesion stand out against normal tissue in general,
                      across slides? This is the quantity a slide-level max/top-K rule
                      depends on.
PER NORMAL SLIDE      accuracy (= specificity) and patches predicted tumor.

OUTPUTS  <out_prefix>_per_slide.csv, _by_bin.csv (mean, median, pooled per witness-rate
         bin), _by_stage.csv, and a 2-panel figure.

SCORES   CONCH from the h5 features (default, GPU), or --score_table with saved margins
         (analyze_zeroshot_score_vs_lesion_size.py --save_patches, FULL table, no --probe).

    python analyze_patch_accuracy_vs_witness_rate.py \
        --xml_dir $SCRATCH/camelyon16/annotations \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --out_prefix res_witness_rate/analysis_patch_wr
"""

import argparse
import os

import numpy as np
import pandas as pd

WR_BINS = [0.0, 0.1, 0.5, 2.0, 10.0, 100.0]          # percent
STAGES = ["ITC", "micro", "macro"]
SLIDE_METRICS = ["accuracy", "balanced_accuracy", "sensitivity", "specificity",
                 "auc_within", "auc_vs_normal_slides"]


def auc(pos, neg):
    """Mann-Whitney AUC with ties at 0.5; NaN if a side is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    s = np.sort(neg)
    lt = np.searchsorted(s, pos, side="left")
    le = np.searchsorted(s, pos, side="right")
    return float(((lt + 0.5 * (le - lt)) / len(neg)).mean())


def rate(num, den):
    return num / den if den else np.nan


def slide_metrics(margin, p, gt, threshold):
    pred = p > threshold
    t, n = gt, ~gt
    tp, fn = int((pred & t).sum()), int((~pred & t).sum())
    tn, fp = int((~pred & n).sum()), int((pred & n).sum())
    sens, spec = rate(tp, tp + fn), rate(tn, tn + fp)
    return {"n_eval_patches": len(gt), "n_tumor_patches": int(t.sum()),
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "accuracy": rate(tp + tn, len(gt)), "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": (sens + spec) / 2,      # NaN if either side is undefined
            "auc_within": auc(margin[t], margin[n])}


def summarise(df, by, order):
    rows = []
    for key in order:
        g = df[df[by] == key]
        r = {by: key, "n_slides": len(g),
             "n_no_tumor_patch": int((g.n_tumor_patches == 0).sum())}   # sensitivity/AUC undefined
        for m in SLIDE_METRICS:
            r[f"{m}_mean"] = g[m].mean() if len(g) else np.nan
            r[f"{m}_median"] = g[m].median() if len(g) else np.nan
        tp, fn, tn, fp = (g[c].sum() for c in ("tp", "fn", "tn", "fp"))
        r["accuracy_pooled"] = rate(tp + tn, tp + fn + tn + fp)
        r["sensitivity_pooled"] = rate(tp, tp + fn)
        r["specificity_pooled"] = rate(tn, tn + fp)
        rows.append(r)
    return pd.DataFrame(rows)


def bin_label(lo, hi):
    return f"{lo:g}-{hi:g}%"


def make_figure(df, wr_col, normal_spec, threshold, gt_desc, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    c1, c2, c3 = "#2a78d6", "#eb6834", "#1baf7a"
    x = df[wr_col].clip(lower=1e-3)
    d = df.assign(_x=x, _bin=pd.cut(df[wr_col], WR_BINS, include_lowest=True))
    med = d.groupby("_bin", observed=True)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.3), sharex=True)
    for ax, series, title, ylabel in [
        (a, [("sensitivity", c2, "sensitivity (tumor patches)"),
             ("specificity", c3, "specificity (normal patches)"),
             ("accuracy", c1, "accuracy (dashed: often on top of specificity)")],
         "Accuracy at P(Tumor) > %g" % threshold, "per-slide value"),
        (b, [("auc_within", c1, "AUC within slide"),
             ("auc_vs_normal_slides", c2, "AUC vs normal-slide patches (dashed)")],
         "Threshold-free separation (AUC)", "AUC"),
    ]:
        for i, (col, color, label) in enumerate(series):
            ok = d[col].notna()
            ax.scatter(d._x[ok], d[col][ok], s=13, color=color, alpha=0.35, linewidth=0, zorder=2)
            m = med.agg(xm=("_x", "median"), ym=(col, "median")).dropna()
            last = i == len(series) - 1          # the series most likely to coincide: on top
            ax.plot(m.xm, m.ym, "o--" if last else "o-", color=color, lw=1.8 if last else 2.2,
                    ms=4 if last else 5.5, mec="white", mew=0.8, label=label,
                    zorder=4 if last else 3)
        if ax is b:
            ax.axhline(0.5, color=ink2, lw=1, ls="--", zorder=1)
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel({"tumor_area_pct": "tumor area, % of tissue",
                       "witness_rate_centre_pct": "witness rate, % of patches (centre rule)"}[wr_col])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", fontsize=10, color=ink)
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.text(0.01, 0.005,
             f"Dots: tumor slides; lines: bin medians.  GT: {gt_desc}.  Normal slides: median "
             f"specificity {normal_spec:.3f}.  At low witness rate accuracy ~ specificity.  "
             f"Witness rate below 0.001% drawn at 0.001%.",
             fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--xml_dir", required=True)
    p.add_argument("--csv_path", default="./local_data/csv/CAMELYON16.csv")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--checkpoint_path",
                   default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--score_table", default=None)
    p.add_argument("--logit_scale", type=float, default=None,
                   help="Only with --score_table (default: CONCH's 56.35).")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--gt", choices=["centre", "occ"], default="centre")
    p.add_argument("--min_occupancy", type=float, default=0.5)
    p.add_argument("--exclude_partial", action="store_true",
                   help="Evaluate only patches with occupancy 0 or >= 0.99.")
    p.add_argument("--wr", choices=["tumor_area_pct", "witness_rate_centre_pct"],
                   default="tumor_area_pct")
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--split", choices=["all", "test", "train"], default="all")
    p.add_argument("--test_prefix", default="test_")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--out_prefix", default="analysis/analysis_patch_wr/zs")
    args = p.parse_args()

    from tqdm import tqdm
    from utils import witness_rate as W
    from utils import zeroshot as Z

    meta = pd.read_csv(args.csv_path)
    if args.split != "all":
        is_test = meta.WSI.astype(str).str.startswith(args.test_prefix)
        meta = meta[is_test if args.split == "test" else ~is_test]
    scorer = Z.make_scorer(args.score_table, args.checkpoint_path, args.text_prototypes,
                           args.logit_scale or Z.CONCH_LOGIT_SCALE, args.batch_size)
    mpp = args.mpp_level0 or W.DEFAULT_MPP_LEVEL0
    gt_desc = ("centre in tumor" if args.gt == "centre"
               else f"occupancy >= {args.min_occupancy:g}") + \
              ("; only clean patches (occupancy 0 or >= 0.99)" if args.exclude_partial else "")

    tumor_rows, tumor_margins, normal_rows, normal_pool = [], {}, [], []
    skipped = {"h5": [], "xml": [], "scores": []}
    for wsi, label in tqdm(list(zip(meta.WSI, meta.GT)), desc="patch metrics"):
        h5 = os.path.join(args.h5_dir, f"{wsi}.h5")
        xml = os.path.join(args.xml_dir, f"{wsi}.xml")
        if not os.path.exists(h5):
            skipped["h5"].append(wsi); continue
        if label == "Tumor" and not os.path.exists(xml):
            skipped["xml"].append(wsi); continue
        grid = W.load_patch_grid(h5, with_features=scorer.needs_features)
        margin = scorer(wsi, grid)
        if margin is None:
            skipped["scores"].append(wsi); continue
        prob = Z.p_tumor_from_margin(margin, scorer.scale)

        if label == "Normal":
            normal_pool.append(margin)
            n_fp = int((prob > args.threshold).sum())
            normal_rows.append({"WSI": wsi, "n_patches": len(margin), "fp": n_fp,
                                "accuracy": 1 - n_fp / len(margin)})
            continue

        ann = W.parse_camelyon16_xml(xml)
        occ = W.patch_occupancy_from_polygons(grid.coords, grid.patch_size, ann)
        centre = W.patch_centre_in_polygons(grid.coords, grid.patch_size, ann)
        gt = centre if args.gt == "centre" else occ >= args.min_occupancy
        keep = ((occ == 0) | (occ >= 0.99)) if args.exclude_partial else np.ones(len(gt), bool)
        rec = {"WSI": wsi, "n_patches": len(margin)}
        rec.update(slide_metrics(margin[keep], prob[keep], gt[keep], args.threshold))
        st = W.slide_witness_stats(occ, centre, args.min_occupancy)
        geo = W.lesion_geometry(ann, mpp)
        rec.update({"tumor_area_pct": st["tumor_area_pct"],
                    "witness_rate_centre_pct": st["witness_rate_centre_pct"],
                    "patch_size_level0": grid.patch_size, "n_lesions": len(ann),
                    "max_lesion_diam_mm": float(geo.diam_max_mm.max()) if len(geo) else np.nan,
                    "lesion_stage": geo.loc[geo.diam_max_mm.idxmax(), "stage"] if len(geo) else "none"})
        tumor_rows.append(rec)
        tumor_margins[wsi] = margin[keep][gt[keep]]

    if not tumor_rows:
        raise SystemExit("no tumor slide evaluated -- check paths / --split")
    df = pd.DataFrame(tumor_rows)
    pool = np.sort(np.concatenate(normal_pool)) if normal_pool else np.array([])
    df["auc_vs_normal_slides"] = [auc(tumor_margins[w], pool) for w in df.WSI]
    labels = [bin_label(lo, hi) for lo, hi in zip(WR_BINS[:-1], WR_BINS[1:])]
    df["wr_bin"] = pd.cut(df[args.wr], WR_BINS, labels=labels, include_lowest=True).astype(str)
    nd = pd.DataFrame(normal_rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    df.to_csv(f"{args.out_prefix}_per_slide.csv", index=False)
    if len(nd):
        nd.to_csv(f"{args.out_prefix}_normal_slides.csv", index=False)
    by_bin = summarise(df, "wr_bin", labels)
    by_stage = summarise(df, "lesion_stage", STAGES)
    by_bin.to_csv(f"{args.out_prefix}_by_bin.csv", index=False)
    by_stage.to_csv(f"{args.out_prefix}_by_stage.csv", index=False)
    normal_spec = float(nd.accuracy.median()) if len(nd) else np.nan
    make_figure(df, args.wr, normal_spec, args.threshold, gt_desc, f"{args.out_prefix}.png")

    cols = ["n_slides", "n_no_tumor_patch"] + [f"{m}_median" for m in SLIDE_METRICS]
    short = {"n_no_tumor_patch": "no_tumor_patch"}
    short.update({f"{m}_median": m.replace("balanced_accuracy", "bal_acc")
                              .replace("auc_vs_normal_slides", "auc_vs_normals")
             for m in SLIDE_METRICS})
    print("\n" + "=" * 104)
    print(f"Zero-shot patch metrics vs witness rate  [{len(df)} tumor / {len(nd)} normal slides, "
          f"P(Tumor) > {args.threshold:g}, GT: {gt_desc}, patch size "
          f"{sorted(df.patch_size_level0.unique().tolist())}]")
    print("=" * 104)
    print(f"medians per slide, by {args.wr} bin:")
    print(by_bin[["wr_bin"] + cols].rename(columns=short)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("-" * 104 + "\nby largest lesion stage:")
    print(by_stage[["lesion_stage"] + cols].rename(columns=short)
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("-" * 104)
    if len(nd):
        print(f"normal slides: median accuracy (= specificity) {normal_spec:.4f}, median "
              f"{nd.fp.median():.0f} patches predicted tumor per slide")
    for k, v in skipped.items():
        if v:
            print(f"skipped, no {k}: {len(v)} ({v[:5]}{'...' if len(v) > 5 else ''})")
    print("=" * 104)
    print(f"-> {args.out_prefix}_per_slide.csv, _by_bin.csv, _by_stage.csv, .png")


if __name__ == "__main__":
    main()
