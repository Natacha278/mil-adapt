"""
analyze_miladapter_attention_vs_witness_rate.py

Few-shot MIL-Adapter (adapter + ABMIL) on CAMELYON16, as a function of each slide's
witness rate, over several seeds. Two things only:

  1. SLIDE-LEVEL ACCURACY
       tumor slides, per witness-rate bin: fraction predicted Tumor (= slide sensitivity)
       normal slides: fraction predicted Normal (= slide specificity)
       overall accuracy and balanced accuracy of the seed's eval split
  2. ABMIL ATTENTION, per tumor slide (attention a_i sums to 1 over the slide's N patches;
     r_i = N * a_i is the RELATIVE attention, 1 = uniform, comparable across slides)
       attention_on_tumor   sum of a_i over the tumor patches: share of the attention
                            budget spent on the lesion
       enrichment           attention_on_tumor / (n_tumor / N): how many times more
                            attention a tumor patch gets than a uniform share
       top1_in_tumor        is the single most-attended patch a tumor patch? (0/1)
       auc_within           AUC of a_i, tumor vs the same slide's normal patches
       auc_vs_normal_slides AUC of r_i, tumor patches vs all patches of the seed's normal
                            eval slides (relative attention, so slide size cancels)

Training is exactly main.py's (same fewshot_sampling split, optimiser, scheduler, loss);
only each seed's EVAL slides are scored (its 16+16 support slides are excluded).
Ground truth and witness rate come from utils/witness_rate.py (true patch footprint from
the h5 coords attributes, "_2" holes subtracted). Witness rate on the x-axis: tumor area
as % of tissue.

AGGREGATION per seed: slide accuracy = mean over the bin's slides; attention metrics =
median over the bin's slides (top1_in_tumor: mean = hit rate). Then mean and SD across
seeds.

OUTPUTS <out_prefix>_per_slide.csv, _by_bin.csv, _by_stage.csv, _per_seed.csv, .png

GPU needed. `--self_test` checks the metric/aggregation/figure code without torch or data.

    python analyze_miladapter_attention_vs_witness_rate.py \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
        --xml_dir $SCRATCH/camelyon16/annotations \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --adapter TaskRes --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out_prefix analysis/analysis_miladapter_attention/TaskRes_k16_descr
"""

import argparse
import os

import numpy as np
import pandas as pd

CLASSES = ["Normal", "Tumor"]
TUMOR = CLASSES.index("Tumor")
WR_BINS = [0.0, 0.1, 0.5, 2.0, 10.0, 100.0]
WR_LABELS = [f"{lo:g}-{hi:g}%" for lo, hi in zip(WR_BINS[:-1], WR_BINS[1:])]
STAGES = ["ITC", "micro", "macro"]
ATT_METRICS = ["attention_on_tumor", "enrichment", "top1_in_tumor", "auc_within",
               "auc_vs_normal_slides"]


# =====================================================================================
# Metrics (numpy only)
# =====================================================================================

def auc(pos, neg_sorted):
    pos = np.asarray(pos, float)
    if len(pos) == 0 or len(neg_sorted) == 0:
        return np.nan
    lt = np.searchsorted(neg_sorted, pos, side="left")
    le = np.searchsorted(neg_sorted, pos, side="right")
    return float(((lt + 0.5 * (le - lt)) / len(neg_sorted)).mean())


def attention_metrics(att, gt, normal_pool_rel):
    """att: ABMIL weights of one slide (sum to 1); gt: bool tumor mask."""
    att = np.asarray(att, float)
    n, k = len(att), int(gt.sum())
    if k == 0:
        return {m: np.nan for m in ATT_METRICS}
    on_tumor = float(att[gt].sum())
    rel = att * n
    return {"attention_on_tumor": on_tumor,
            "enrichment": on_tumor / (k / n),
            "top1_in_tumor": float(gt[int(np.argmax(att))]),
            "auc_within": auc(att[gt], np.sort(att[~gt])) if k < n else np.nan,
            "auc_vs_normal_slides": auc(rel[gt], normal_pool_rel)}


def seed_records(seed, scored, gt_cache, labels):
    """scored: {wsi: {"att": weights or None, "slide_p": float}}."""
    pool = [s["att"] * len(s["att"]) for w, s in scored.items()
            if labels[w] == "Normal" and s["att"] is not None]
    pool = np.sort(np.concatenate(pool)) if pool else np.array([])
    rows = []
    for w, s in scored.items():
        pred = CLASSES[int(s["slide_p"] > 0.5)]
        r = {"seed": seed, "WSI": w, "gt_label": labels[w], "slide_pred": pred,
             "slide_p_tumor": s["slide_p"], "slide_correct": float(pred == labels[w])}
        if labels[w] == "Tumor":
            g = gt_cache[w]
            r.update({k: g[k] for k in ("tumor_area_pct", "witness_rate_centre_pct",
                                        "lesion_stage", "max_lesion_diam_mm")})
            r["n_patches"] = len(g["gt"])
            r["n_tumor_patches"] = int(g["gt"].sum())
            if s["att"] is not None:
                r.update(attention_metrics(s["att"], g["gt"], pool))
        rows.append(r)
    return rows


def per_seed_summary(df):
    out = []
    for seed, g in df.groupby("seed"):
        t, n = g[g.gt_label == "Tumor"], g[g.gt_label == "Normal"]
        sens, spec = t.slide_correct.mean(), n.slide_correct.mean()
        out.append({"seed": seed, "n_tumor": len(t), "n_normal": len(n),
                    "accuracy": g.slide_correct.mean(), "slide_sensitivity": sens,
                    "slide_specificity": spec, "balanced_accuracy": (sens + spec) / 2})
    return pd.DataFrame(out)


def aggregate(df, by, order):
    t = df[df.gt_label == "Tumor"]
    rows = []
    for key in order:
        g = t[t[by] == key]
        ps = g.groupby("seed").agg(
            slide_accuracy=("slide_correct", "mean"), n=("WSI", "size"),
            **{m: (m, "mean" if m == "top1_in_tumor" else "median") for m in ATT_METRICS})
        r = {by: key, "n_slides_per_seed": ps.n.mean() if len(ps) else 0}
        for m in ["slide_accuracy"] + ATT_METRICS:
            r[f"{m}_mean"] = ps[m].mean() if len(ps) else np.nan
            r[f"{m}_sd"] = ps[m].std() if len(ps) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# =====================================================================================
# Figure
# =====================================================================================

def make_figure(by_bin, x_of_bin, per_seed, title, footer, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"slide_accuracy": "#2a78d6", "auc_within": "#2a78d6",
              "auc_vs_normal_slides": "#eb6834", "top1_in_tumor": "#1baf7a",
              "attention_on_tumor": "#4a3aa7"}
    names = {"slide_accuracy": "tumor slides predicted Tumor",
             "auc_within": "AUC within slide", "auc_vs_normal_slides": "AUC vs normal-slide patches",
             "top1_in_tumor": "top-1 attended patch is tumor (rate)",
             "attention_on_tumor": "share of attention on the lesion"}
    xs = np.array([x_of_bin.get(b, np.nan) for b in WR_LABELS])
    d = by_bin.set_index("wr_bin").reindex(WR_LABELS)

    def line(ax, m):
        y = d[f"{m}_mean"].to_numpy(float)
        e = np.nan_to_num(d[f"{m}_sd"].to_numpy(float))
        ok = np.isfinite(xs) & np.isfinite(y)
        ax.fill_between(xs[ok], (y - e)[ok], (y + e)[ok], color=colors[m], alpha=0.13, lw=0)
        ax.plot(xs[ok], y[ok], "o-", color=colors[m], lw=2.2, ms=5.5, mec="white", mew=0.8,
                label=names[m], zorder=3)

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    line(a, "slide_accuracy")
    spec_m, spec_s = per_seed.slide_specificity.mean(), per_seed.slide_specificity.std()
    a.axhspan(spec_m - spec_s, spec_m + spec_s, color=ink2, alpha=0.10, lw=0)
    a.axhline(spec_m, color=ink2, lw=1.4, ls="--",
              label=f"normal slides predicted Normal ({spec_m:.2f})")
    for m in ["auc_within", "auc_vs_normal_slides", "top1_in_tumor", "attention_on_tumor"]:
        line(b, m)
    b.axhline(0.5, color=ink2, lw=1, ls=":", zorder=1)
    for ax, t, yl in [(a, "Slide-level accuracy", "accuracy (mean ± SD over seeds)"),
                      (b, "ABMIL attention on the lesion", "per-slide median (mean ± SD over seeds)")]:
        ax.set_xscale("log")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("tumor area, % of tissue")
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left", fontsize=10, color=ink)
        ax.grid(color=grid, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=ink)
    fig.text(0.01, 0.005, footer, fontsize=7, color=ink2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================================================
# Self-test (numpy + matplotlib only)
# =====================================================================================

def self_test(out_dir):
    rng = np.random.default_rng(0)
    # hand check of the attention metrics on a tiny slide
    att = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
    gt = np.array([True, False, True, False, False, False])
    pool = np.sort(np.array([0.5, 1.0, 1.0, 2.0]))
    m = attention_metrics(att, gt, pool)
    assert abs(m["attention_on_tumor"] - 0.5) < 1e-12
    assert abs(m["enrichment"] - 0.5 / (2 / 6)) < 1e-12
    assert m["top1_in_tumor"] == 1.0
    assert abs(m["auc_within"] - (4 + 2.5) / 8) < 1e-12        # 0.4 beats all 4; 0.1 beats 2, ties 1
    assert abs(m["auc_vs_normal_slides"] - (1.0 + 0.25) / 2) < 1e-12  # rel 2.4 beats all; 0.6 beats 1/4
    print("  hand-checked attention metrics: OK")

    labels, gt_cache = {}, {}
    wr = 10 ** rng.uniform(-2.5, 1.3, 50)
    for i in range(80):
        w = f"s{i}"
        labels[w] = "Tumor" if i < 50 else "Normal"
        if i < 50:
            n = 3000; k = max(1, int(n * wr[i] / 100))
            g = np.zeros(n, bool); g[:k] = True
            gt_cache[w] = {"gt": g, "tumor_area_pct": wr[i], "witness_rate_centre_pct": wr[i],
                           "max_lesion_diam_mm": 1.0,
                           "lesion_stage": "ITC" if wr[i] < 0.05 else "micro" if wr[i] < 2 else "macro"}
    rows = []
    for seed in range(4):
        scored = {}
        for w, lab in labels.items():
            n = 3000
            g = gt_cache[w]["gt"] if lab == "Tumor" else np.zeros(n, bool)
            logit = rng.standard_normal(n) + 2.5 * g
            a = np.exp(logit); a /= a.sum()
            frac = g.mean()
            p = 1 / (1 + np.exp(-(np.log10(frac + 1e-5) + 2.2) * 3)) if lab == "Tumor" else rng.uniform(0, 0.6)
            scored[w] = {"att": a, "slide_p": float(p)}
        rows += seed_records(seed, scored, gt_cache, labels)
    df = pd.DataFrame(rows)
    t = df.gt_label == "Tumor"
    df.loc[t, "wr_bin"] = pd.cut(df.loc[t, "tumor_area_pct"], WR_BINS, labels=WR_LABELS,
                                 include_lowest=True).astype(str)
    by_bin = aggregate(df, "wr_bin", WR_LABELS)
    ps = per_seed_summary(df)
    acc = by_bin.slide_accuracy_mean.dropna().to_numpy()
    assert acc[-1] > acc[0], "synthetic slide accuracy should rise with witness rate"
    x = df[t].groupby("wr_bin").tumor_area_pct.median().to_dict()
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, "miladapter_attention_self_test.png")
    make_figure(by_bin, x, ps, "self-test: synthetic attention and predictions, 4 seeds",
                "synthetic", png)
    print(by_bin[["wr_bin", "n_slides_per_seed", "slide_accuracy_mean"] +
                 [f"{m}_mean" for m in ATT_METRICS]].to_string(index=False,
                                                               float_format=lambda v: f"{v:.3f}"))
    print(f"self_test: OK ({png})")


# =====================================================================================
# Main
# =====================================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", help="MIL-format data root (folder/<project>/<encoder>/*.npy)")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--xml_dir")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--adapter", default="TaskRes",
                   choices=["ZSMIL", "TaskRes", "CLIPAdapter", "TIPAdapter"])
    p.add_argument("--init", choices=["ZS", "random"], default="ZS", help="ZSMIL only.")
    p.add_argument("--k_shots", type=int, default=16, choices=[2, 4, 8, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gt", choices=["centre", "occ"], default="centre")
    p.add_argument("--min_occupancy", type=float, default=0.5)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--out_prefix", default="analysis/analysis_miladapter_attention/miladapter")
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test(os.path.dirname(os.path.abspath(args.out_prefix)))
        return
    if not args.folder or not args.xml_dir:
        p.error("--folder and --xml_dir are required (or use --self_test)")

    import torch
    from tqdm import tqdm
    from utils.trainer import train_model
    from utils.utils import set_random_seeds, load_data, fewshot_sampling
    from utils import witness_rate as W
    from visu_ABMIL import build_model, patch_known_bugs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, WSI = load_data(folder=args.folder, project=args.project, encoder=args.encoder,
                          classes=CLASSES)
    labels = {w: CLASSES[y] for w, y in zip(WSI, Y)}
    text_np = np.load(args.text_prototypes)
    mpp = args.mpp_level0 or W.DEFAULT_MPP_LEVEL0

    gt_cache, skipped = {}, []
    for i, w in enumerate(tqdm(WSI, desc="ground truth")):
        if labels[w] != "Tumor":
            continue
        h5 = os.path.join(args.h5_dir, f"{w}.h5")
        xml = os.path.join(args.xml_dir, f"{w}.xml")
        if not (os.path.exists(h5) and os.path.exists(xml)):
            skipped.append(w)
            continue
        grid = W.load_patch_grid(h5)
        if len(grid) != len(X[i]):
            raise SystemExit(f"{w}: {len(X[i])} patches in the .npy but {len(grid)} in the .h5")
        ann = W.parse_camelyon16_xml(xml)
        occ = W.patch_occupancy_from_polygons(grid.coords, grid.patch_size, ann)
        cen = W.patch_centre_in_polygons(grid.coords, grid.patch_size, ann)
        st = W.slide_witness_stats(occ, cen, args.min_occupancy)
        geo = W.lesion_geometry(ann, mpp)
        gt_cache[w] = {
            "gt": cen if args.gt == "centre" else occ >= args.min_occupancy,
            "tumor_area_pct": st["tumor_area_pct"],
            "witness_rate_centre_pct": st["witness_rate_centre_pct"],
            "max_lesion_diam_mm": float(geo.diam_max_mm.max()) if len(geo) else np.nan,
            "lesion_stage": geo.loc[geo.diam_max_mm.idxmax(), "stage"] if len(geo) else "none"}

    rows = []
    for seed in args.seeds:
        set_random_seeds(seed_value=seed)
        train_data, val_data, train_labels, _ = fewshot_sampling(X=X, Y=Y, k_shots=args.k_shots,
                                                                 seed=seed)
        id_to_idx = {id(x): i for i, x in enumerate(X)}
        val_idx = [id_to_idx[id(v)] for v in val_data]
        model = build_model(args.adapter, text_np, "ABMIL", args.init,
                            train_data, train_labels).to(device)
        model = patch_known_bugs(model, args.adapter, "ABMIL")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        train_model(model, opt, torch.nn.CrossEntropyLoss(reduction="sum"), sch,
                    train_data, train_labels, args.epochs)
        model.eval()
        scored = {}
        with torch.no_grad():
            for i in tqdm(val_idx, desc=f"seed {seed}"):
                w = WSI[i]
                if labels[w] == "Tumor" and w not in gt_cache:
                    continue
                H = torch.as_tensor(X[i], dtype=torch.float32, device=device)
                slide_p = float(model(H)[0].softmax(dim=0)[TUMOR])
                att = model.ABMIL(H)[1].squeeze(-1).cpu().numpy()
                scored[w] = {"att": att, "slide_p": slide_p}
        rows += seed_records(seed, scored, gt_cache, labels)

    df = pd.DataFrame(rows)
    t = df.gt_label == "Tumor"
    df.loc[t, "wr_bin"] = pd.cut(df.loc[t, "tumor_area_pct"], WR_BINS, labels=WR_LABELS,
                                 include_lowest=True).astype(str)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    df.to_csv(f"{args.out_prefix}_per_slide.csv", index=False)
    ps = per_seed_summary(df)
    ps.to_csv(f"{args.out_prefix}_per_seed.csv", index=False)
    by_bin = aggregate(df, "wr_bin", WR_LABELS)
    by_stage = aggregate(df, "lesion_stage", STAGES)
    by_bin.to_csv(f"{args.out_prefix}_by_bin.csv", index=False)
    by_stage.to_csv(f"{args.out_prefix}_by_stage.csv", index=False)

    x_of_bin = df[t].groupby("wr_bin").tumor_area_pct.median().to_dict()
    gt_desc = "centre in tumor" if args.gt == "centre" else f"occupancy >= {args.min_occupancy:g}"
    title = (f"{args.adapter}+ABMIL, k={args.k_shots}, {len(args.seeds)} seeds, eval slides only: "
             f"accuracy {ps.accuracy.mean():.3f} ± {ps.accuracy.std():.3f}, balanced "
             f"{ps.balanced_accuracy.mean():.3f} ± {ps.balanced_accuracy.std():.3f}")
    footer = (f"Lines: mean over seeds (slide accuracy: per-bin mean; attention: per-bin median); "
              f"bands: ± SD across seeds.  Patch GT: {gt_desc}.  Prompts: "
              f"{os.path.basename(args.text_prototypes)}.")
    make_figure(by_bin, x_of_bin, ps, title, footer, f"{args.out_prefix}.png")

    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    cols = ["wr_bin", "n_slides_per_seed", "slide_accuracy_mean", "slide_accuracy_sd"] + \
           [f"{m}_mean" for m in ATT_METRICS]
    print(by_bin[cols].rename(columns=lambda c: c.replace("_mean", "")
                              .replace("auc_vs_normal_slides", "auc_vs_normals"))
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("-" * 100)
    print(f"normal slides predicted Normal: {ps.slide_specificity.mean():.3f} ± "
          f"{ps.slide_specificity.std():.3f}  |  tumor slides predicted Tumor: "
          f"{ps.slide_sensitivity.mean():.3f} ± {ps.slide_sensitivity.std():.3f}")
    if skipped:
        print(f"tumor slides without h5/xml, skipped: {len(skipped)} ({skipped[:5]})")
    print("=" * 100)
    print(f"-> {args.out_prefix}_per_slide.csv, _per_seed.csv, _by_bin.csv, _by_stage.csv, .png")


if __name__ == "__main__":
    main()
