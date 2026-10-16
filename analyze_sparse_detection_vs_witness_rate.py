"""
analyze_sparse_detection_vs_witness_rate.py

Does adaptive sparse detection (utils/sparse_detection.py: patch p-values against normal
patches + Higher Criticism / Berk-Jones) beat fixed top-K pooling, by witness rate?

Pooling is the only thing that changes: every rule sees the SAME patch scores.
  scorers   zero-shot   alpha * cos(h, w0)  (text direction, no training)
            learned     Text-TopK (utils/text_topk_mil_torch.py) trained on the support
  rules     top-1, top-K (--ks), mean          on raw patch scores
            min-p, HC, HC+, BJ                 on patch p-values

Null for the p-values (per seed, never eval slides): every patch of the seed's normal
SUPPORT slides. For the learned scorer, --learned_null
  heldout   (default) the support normals are split in two halves; the model is trained
            on one half + all support tumors, the other half's patches are the null.
            Clean (the null was never trained on), at the cost of a model trained on
            k/2 normals.
  insample  model trained on the whole support; null = its own training normals
            (optimistic: training pushed exactly those patches down).

Evaluation per seed on the eval slides, then mean ± SD over seeds, by tumor-area bin
and lesion stage:
  auc          tumor slides of the bin vs ALL normal eval slides (threshold-free)
  sens@spec    tumor slides above the --spec quantile of the normal eval slides'
               statistic (a common operating point to compare rules; not a deployable
               threshold -- that is the conformal step)

OUTPUTS <out_prefix>_per_slide.csv (one row per slide x scorer x rule), _by_bin.csv,
        _by_stage.csv, _overall.csv, _by_witness_rate.png, _by_stage.png

    python analyze_sparse_detection_vs_witness_rate.py --self_test
    python analyze_sparse_detection_vs_witness_rate.py \\
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \\
        --xml_dir $SCRATCH/camelyon16/annotations \\
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \\
        --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --out_prefix analysis/analysis_sparse/sparse_k16
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_text_topk_vs_witness_rate import (CLASSES, STAGES, WR_BINS, WR_LABELS,  # noqa: E402
                                               frac_below, ground_truth)
from utils.sparse_detection import all_statistics                                  # noqa: E402

SCORERS = ["zero-shot", "learned"]
SPARSE_RULES = ["HC", "BJ"]                       # plotted; HC+ and min-p go to the CSVs
# fixed categorical order (validated palette: blue, orange, aqua, yellow, magenta)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
MARKERS = ["o", "s", "^", "D", "v"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_rules(ks):
    return [f"top-{k:g}" for k in ks] + ["mean"] + SPARSE_RULES


# =====================================================================================
# Scoring
# =====================================================================================

def score_seed(seed, X, Y, wsis, protos, args, device):
    import torch
    import torch.nn.functional as F
    from utils import tumor_ratio_metrics as TRM
    from utils.text_topk_mil_torch import TextTopKMIL

    tr, va = TRM.fewshot_split_ids(Y, args.k_shots, seed)
    ref = TextTopKMIL(protos, k=args.topk, lam=args.lam).to(device)
    w0, scale = ref.w0, ref.scale

    def zs(H):
        with torch.no_grad():
            Hn = F.normalize(torch.as_tensor(H, device=device).to(w0.dtype), dim=-1)
            return (scale * (Hn @ w0)).cpu().numpy().astype(np.float64)

    scorers = {"zero-shot": zs}
    nulls = {"zero-shot": np.sort(np.concatenate([zs(X[i]) for i in tr[Y[tr] == 0]]))}
    info = {"seed": seed}
    if "learned" in args.scorers:
        normals = tr[Y[tr] == 0]
        if args.learned_null == "heldout":
            perm = np.random.default_rng(1000 + seed).permutation(normals)
            null_ids, fit_normals = perm[: len(perm) // 2], perm[len(perm) // 2:]
        else:
            null_ids, fit_normals = normals, normals
        fit_ids = np.concatenate([fit_normals, tr[Y[tr] == 1]])
        model = TextTopKMIL(protos, k=args.topk, lam=args.lam).to(device)
        extra = {"neg_weight": args.neg_weight} if args.neg_weight > 0 else {}
        model.fit([X[i] for i in fit_ids], Y[fit_ids], epochs=args.epochs, lr=args.lr,
                  lr_b=args.lr_b, **extra)

        def learned(H, m=model):
            with torch.no_grad():
                return m.patch_scores(torch.as_tensor(H, device=device)).cpu().numpy().astype(np.float64)

        scorers["learned"] = learned
        nulls["learned"] = np.sort(np.concatenate([learned(X[i]) for i in null_ids]))
        info.update({"learned_null": args.learned_null, "n_null_slides": len(null_ids),
                     "n_fit_normals": len(fit_normals), **model.summary()})
    scorers = {k: v for k, v in scorers.items() if k in args.scorers}
    rows = []
    for i in va:
        for name, f in scorers.items():
            st = all_statistics(f(X[i]), nulls[name], ks=args.ks, alpha0=args.alpha0)
            rows += [{"seed": seed, "WSI": wsis[i], "gt_label": CLASSES[Y[i]], "scorer": name,
                      "rule": r, "value": v} for r, v in st.items()]
    info["n_null_patches_zero_shot"] = len(nulls["zero-shot"])
    return rows, info


# =====================================================================================
# Metrics
# =====================================================================================

def evaluate(df, spec):
    """Per (seed, scorer, rule): slide placement vs normal eval slides, and detection at
    the spec-quantile of the normal eval slides' statistic."""
    out = []
    for (seed, sc, rule), g in df.groupby(["seed", "scorer", "rule"]):
        n = np.sort(g[g.gt_label == "Normal"].value.values)
        thr = n[min(len(n) - 1, int(np.ceil(spec * len(n))) - 1)] if len(n) else np.inf
        t = g[g.gt_label == "Tumor"]
        out.append(pd.DataFrame({"seed": seed, "scorer": sc, "rule": rule, "WSI": t.WSI.values,
                                 "placement": frac_below(n, t.value.values) if len(n) else np.nan,
                                 "detected": (t.value.values > thr).astype(float),
                                 "spec_achieved": float((n <= thr).mean()) if len(n) else np.nan}))
    return pd.concat(out, ignore_index=True)


def aggregate(ev, by, order):
    rows = []
    for (sc, rule), g in ev.groupby(["scorer", "rule"]):
        for key in order:
            ps = g[g[by] == key].groupby("seed").agg(auc=("placement", "mean"),
                                                    sens=("detected", "mean"), n=("WSI", "size"))
            rows.append({by: key, "scorer": sc, "rule": rule,
                         "n_slides_per_seed": ps.n.mean() if len(ps) else 0,
                         "auc_mean": ps.auc.mean(), "auc_sd": ps.auc.std(),
                         "sens_mean": ps.sens.mean(), "sens_sd": ps.sens.std()})
    return pd.DataFrame(rows)


def overall(ev):
    ps = ev.groupby(["scorer", "rule", "seed"]).agg(auc=("placement", "mean"),
                                                    sens=("detected", "mean")).reset_index()
    return ps.groupby(["scorer", "rule"]).agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                                              sens_mean=("sens", "mean"),
                                              sens_sd=("sens", "std")).reset_index()


# =====================================================================================
# Figure
# =====================================================================================

def make_figure(tab, key, order, scorers, rules, xlabel, title, footer, spec, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    x = np.arange(len(order))
    fig, axes = plt.subplots(len(scorers), 2, figsize=(13, 3.9 * len(scorers) + 0.9),
                             squeeze=False)
    for r_i, sc in enumerate(scorers):
        for c_i, (metric, ylabel) in enumerate((("auc", "slide AUC vs normal slides"),
                                                (f"sens", f"tumor slides detected @ spec {spec:g}"))):
            ax = axes[r_i, c_i]
            for j, rule in enumerate(rules):
                d = tab[(tab.scorer == sc) & (tab.rule == rule)].set_index(key).reindex(order)
                y = d[f"{metric}_mean"].to_numpy(float)
                e = np.nan_to_num(d[f"{metric}_sd"].to_numpy(float))
                ok = np.isfinite(y)
                sparse = rule in SPARSE_RULES
                ax.fill_between(x[ok], (y - e)[ok], np.minimum(1, (y + e)[ok]),
                                color=COLORS[j], alpha=0.10, lw=0)
                ax.plot(x[ok], y[ok], "-" if sparse else "--", marker=MARKERS[j], color=COLORS[j],
                        lw=2.0, ms=6, mec="white", mew=0.8, label=rule)
            n = tab[(tab.scorer == sc) & (tab.rule == rules[0])].set_index(key).reindex(order).n_slides_per_seed
            ax.set_xticks(x)
            ax.set_xticklabels([f"{g}\n(n≈{v:.0f})" for g, v in zip(order, n.fillna(0))])
            ax.set_ylim(-0.02, 1.02)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            ax.set_title(f"{sc} patch scores — {ylabel}", loc="left", fontsize=10, color=INK)
            if r_i == len(scorers) - 1:
                ax.set_xlabel(xlabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(rules), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.955), title="slide rule (dashed: fixed, solid: adaptive)",
               title_fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    import textwrap
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(footer, 190)), fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.915))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================================================
# Pipeline
# =====================================================================================

def run(args):
    import torch
    from tqdm import tqdm
    from utils import witness_rate as W
    meta = pd.read_csv(args.csv_path)
    wsis = meta.WSI.to_numpy()
    labels = dict(zip(meta.WSI, meta.GT))
    Y = np.array([CLASSES.index(g) for g in meta.GT])
    X = [np.load(os.path.join(args.folder, args.project, args.encoder, f"{w}.npy")).astype(np.float32)
         for w in tqdm(wsis, desc="loading features")]
    gt, skipped = ground_truth(wsis, labels, X, args.h5_dir, args.xml_dir, "centre", 0.5,
                               args.mpp_level0 or W.DEFAULT_MPP_LEVEL0)
    protos = np.load(args.text_prototypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, infos = [], []
    for seed in tqdm(args.seeds, desc="seeds"):
        r, info = score_seed(seed, X, Y, wsis, protos, args, device)
        rows += r
        infos.append(info)
    df = pd.DataFrame(rows)
    df = df[(df.gt_label == "Normal") | df.WSI.isin(gt.keys())]       # tumor slides need GT
    meta_t = pd.DataFrame([{"WSI": w, "tumor_area_pct": g["tumor_area_pct"],
                            "lesion_stage": g["lesion_stage"]} for w, g in gt.items()])
    return df, meta_t, pd.DataFrame(infos), skipped


def report(df, meta_t, infos, skipped, args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    df.to_csv(f"{args.out_prefix}_per_slide.csv", index=False)
    infos.to_csv(f"{args.out_prefix}_per_seed.csv", index=False)
    ev = evaluate(df, args.spec).merge(meta_t, on="WSI", how="left")
    ev["wr_bin"] = pd.cut(ev.tumor_area_pct, WR_BINS, labels=WR_LABELS, include_lowest=True).astype(str)
    by_bin, by_stage, ov = aggregate(ev, "wr_bin", WR_LABELS), aggregate(ev, "lesion_stage", STAGES), overall(ev)
    by_bin.to_csv(f"{args.out_prefix}_by_bin.csv", index=False)
    by_stage.to_csv(f"{args.out_prefix}_by_stage.csv", index=False)
    ov.to_csv(f"{args.out_prefix}_overall.csv", index=False)

    scorers = [s for s in SCORERS if s in args.scorers]
    rules = plot_rules(args.ks)
    tag = (f"k={args.k_shots}, {len(args.seeds)} seeds"
           + (f", learned null: {args.learned_null}" if "learned" in scorers else ""))
    footer = (f"Same patch scores for every rule. p-values against all patches of the seed's normal support "
              f"slides{' (learned: held-out half)' if args.learned_null == 'heldout' else ''}; HC/BJ searched "
              f"over the top {args.alpha0:g} of each slide. Detection threshold = {args.spec:g} quantile of the "
              f"normal eval slides (comparison point, not a deployable threshold). Eval slides only.")
    make_figure(by_bin, "wr_bin", WR_LABELS, scorers, rules, "tumor area, % of tissue",
                f"Slide pooling rules by witness rate ({tag})", footer, args.spec,
                f"{args.out_prefix}_by_witness_rate.png")
    make_figure(by_stage, "lesion_stage", STAGES, scorers, rules,
                "largest lesion (ITC < 0.2 mm ≤ micro < 2 mm ≤ macro)",
                f"Slide pooling rules by lesion stage ({tag})", footer, args.spec,
                f"{args.out_prefix}_by_stage.png")

    pd.set_option("display.width", 220)
    for sc in scorers:
        print("\n" + "=" * 100)
        o = ov[ov.scorer == sc].set_index("rule")
        print(f"{sc} patch scores — overall: " + ", ".join(
            f"{r} AUC {o.loc[r, 'auc_mean']:.3f} / sens {o.loc[r, 'sens_mean']:.3f}" for r in o.index))
        for name, tab, key, order in (("tumor area", by_bin, "wr_bin", WR_LABELS),
                                      ("lesion stage", by_stage, "lesion_stage", STAGES)):
            t = tab[tab.scorer == sc]
            auc = t.pivot(index=key, columns="rule", values="auc_mean").reindex(order)
            sens = t.pivot(index=key, columns="rule", values="sens_mean").reindex(order)
            print(f"-- by {name}: slide AUC | sens @ spec {args.spec:g}")
            print(pd.concat({"AUC": auc, f"sens@{args.spec:g}": sens}, axis=1)
                  .to_string(float_format=lambda v: f"{v:.3f}"))
    if skipped:
        print(f"\ntumor slides without h5/xml, skipped: {len(skipped)}")
    print(f"\n-> {args.out_prefix}_by_witness_rate.png, _by_stage.png, _per_slide.csv, _by_bin.csv, "
          f"_by_stage.csv, _overall.csv, _per_seed.csv")
    return ov, by_bin


def self_test(out_dir):
    from analyze_text_topk_vs_witness_rate import make_synthetic_cohort   # self-test only
    # cohort with many look-alikes and lesions from ~1 patch to ~17% of the tissue
    root = make_synthetic_cohort(seed=7, n_slides=96, radii=(500, 800, 1500, 3000, 6000, 9000),
                                 n_mimic=6)
    args = argparse.Namespace(
        folder=os.path.join(root, "feat"), project="CAMELYON16", encoder="CONCH",
        csv_path=os.path.join(root, "C.csv"), h5_dir=os.path.join(root, "h5"),
        xml_dir=os.path.join(root, "xml"), text_prototypes=os.path.join(root, "protos.npy"),
        k_shots=8, seeds=[0, 1, 2], scorers=SCORERS, learned_null="heldout", topk=1, lam=0.1,
        epochs=150, lr=5e-3, lr_b=5e-2, neg_weight=0.0, ks=(1, 10), alpha0=0.5, spec=0.9,
        mpp_level0=None, out_prefix=os.path.join(out_dir, "sparse_self_test"))
    df, meta_t, infos, skipped = run(args)
    assert not skipped and set(df.scorer) == set(SCORERS)
    assert (infos.n_null_slides == 4).all() and (infos.n_fit_normals == 4).all()
    ov, by_bin = report(df, meta_t, infos, skipped, args)
    # every rule sees the same slides; placement in [0, 1]; spec achieved ~ target
    ev = evaluate(df, args.spec)
    assert ev.placement.between(0, 1).all() and (ev.spec_achieved >= args.spec - 1e-9).all()
    assert ev.groupby(["seed", "scorer", "rule"]).size().nunique() == 1
    # the insample null is exercised too
    args2 = argparse.Namespace(**{**vars(args), "learned_null": "insample", "seeds": [0],
                                  "scorers": ["learned"]})
    df2, _, infos2, _ = run(args2)
    assert (infos2.n_null_slides == 8).all() and set(df2.scorer) == {"learned"}
    print("self_test: OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--csv_path", default="./local_data/csv/CAMELYON16.csv")
    p.add_argument("--h5_dir", default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                                       "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--xml_dir")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--k_shots", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--scorers", nargs="+", choices=SCORERS, default=SCORERS)
    p.add_argument("--learned_null", choices=["heldout", "insample"], default="heldout")
    p.add_argument("--topk", type=float, default=1, help="Text-TopK training K (learned scorer).")
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_b", type=float, default=5e-2)
    p.add_argument("--neg_weight", type=float, default=0.0)
    p.add_argument("--ks", type=float, nargs="+", default=[1, 10], help="Fixed top-K rules to compare.")
    p.add_argument("--alpha0", type=float, default=0.5,
                   help="HC/BJ search the most suspicious alpha0 fraction of each slide's patches.")
    p.add_argument("--spec", type=float, default=0.95)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--out_prefix", default="analysis/analysis_sparse/sparse")
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()
    args.ks = tuple(args.ks)
    if args.self_test:
        self_test(os.path.dirname(os.path.abspath(args.out_prefix)))
        return
    if not args.folder or not args.xml_dir:
        p.error("--folder and --xml_dir are required (or use --self_test)")
    report(*run(args), args)


if __name__ == "__main__":
    main()
