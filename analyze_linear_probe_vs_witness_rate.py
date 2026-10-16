"""
analyze_linear_probe_vs_witness_rate.py

Ceiling analysis for low-witness-rate detection on frozen CONCH features, in the k-shot
setting: how far is each scorer from what the ENCODER allows, by witness rate?

Per seed and k (same k-per-class support draw as main.py / MIL-Adapter), four patch
scorers are applied to the same eval patches:
  zero-shot        alpha * cos(h, w0)  (text direction, no training)
  text-topk        Text-TopK MIL trained with SLIDE labels only (utils/text_topk_mil_torch.py;
                   --topk / --lam / --neg_weight as in run_text_topk_mil.py)
  wr-pseudolabel   Text-TopK + witness-rate-aware pseudo-labels (utils/witness_pseudolabel.py):
                   still SLIDE labels only; patch labels estimated from the excess of small
                   p-values vs normal tissue (--pl_rounds / --pl_patch_weight / --pl_z)
  oracle-linear    logistic regression on PATCH labels of the support slides (oracle):
                   positives = annotated tumor patches of the support tumor slides,
                   negatives = all patches of the support normal slides + the non-tumor
                   patches of the support tumor slides. Best any LINEAR scorer can do with
                   this support set -> ceiling for Text-TopK's model class.
  oracle-mlp       small MLP on the same patch labels -> rough ENCODER ceiling.

Reading the ladder (per tumor-area bin):
  text-topk  -> oracle-linear : what a better linear scorer could still recover
  wr-pseudolabel              : how much of that gap estimated patch labels close
  oracle-linear -> oracle-mlp : what a nonlinear scorer could add
  below oracle-mlp            : beyond what these frozen features give at this k

Slide score for every scorer: top-K mean of patch scores (--pool_k, default 1: at low
witness rate the max is the natural rule, so the comparison isolates the scorer).

Metrics on eval slides (support slides excluded), per tumor slide, then by bin / stage:
  best_beats_normal_max   share of normal eval slides whose highest patch is below this
                          slide's best tumor patch (the quantity top-1 detection rests on)
  tpr_at_fpr              tumor patches above the (1 - --fpr) quantile of normal patches
  auc_within              tumor vs the slide's own normal patches (bulk separation)
  slide_auc               slide score vs all normal eval slides (placement)
  detected                slide score above the --spec quantile of normal eval slides
Aggregation: mean per seed within a bin, then mean ± SD over seeds.

OUTPUTS <out_prefix>_per_slide.csv, _by_bin.csv, _by_stage.csv, _per_seed.csv,
        _by_witness_rate.png, _by_stage.png  (rows: metrics, columns: k)

    python analyze_linear_probe_vs_witness_rate.py --self_test
    python analyze_linear_probe_vs_witness_rate.py \\
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \\
        --xml_dir $SCRATCH/camelyon16/annotations \\
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \\
        --k_shots 4 8 16 --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --topk 3 --lam 0.1 --neg_weight 1 \\
        --out_prefix analysis/analysis_ceiling/ceiling
"""

import argparse
import os
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_text_topk_vs_witness_rate import (CLASSES, STAGES, WR_BINS, WR_LABELS,  # noqa: E402
                                               frac_below, ground_truth)

SCORERS = ["zero-shot", "text-topk", "wr-pseudolabel", "oracle-linear", "oracle-mlp"]
METRICS = [("best_beats_normal_max", "best tumor patch beats normal-slide max"),
           ("slide_auc", "slide AUC vs normal slides"),
           ("detected", "tumor slides detected @ spec"),
           ("tpr_at_fpr", "tumor patches above normal FPR quantile"),
           ("auc_within", "patch AUC within slide (bulk)")]
PLOT_METRICS = ["best_beats_normal_max", "slide_auc", "detected"]
# fixed categorical order (validated palette): blue, orange, aqua, yellow
COLORS = {"zero-shot": "#2a78d6", "text-topk": "#eb6834", "oracle-linear": "#1baf7a",
          "oracle-mlp": "#eda100",
          "wr-pseudolabel": "#6250d6"}
MARKERS = {"zero-shot": "o", "text-topk": "s", "wr-pseudolabel": "v", "oracle-linear": "^", "oracle-mlp": "D"}
STYLES = {"zero-shot": "-", "text-topk": "-", "wr-pseudolabel": "-", "oracle-linear": "--", "oracle-mlp": "--"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# =====================================================================================
# Oracle patch classifiers
# =====================================================================================

def patch_training_set(ids, Y, X, gt, max_neg, rng):
    """Patch-labelled set from support slides. Tumor slides without GT are skipped."""
    pos, neg = [], []
    for i in ids:
        H = X[i]
        if Y[i] == 0:
            neg.append(H)
        else:
            g = gt.get(i)
            if g is None:
                continue
            pos.append(H[g])
            neg.append(H[~g])
    P = np.concatenate(pos) if pos else np.zeros((0, X[0].shape[1]), np.float32)
    N = np.concatenate(neg)
    if max_neg and len(N) > max_neg:
        N = N[rng.choice(len(N), max_neg, replace=False)]
    return P, N


def fit_linear(P, N, C, seed):
    from sklearn.linear_model import LogisticRegression
    Xtr = np.concatenate([P, N]).astype(np.float32)
    ytr = np.concatenate([np.ones(len(P)), np.zeros(len(N))])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=seed)
    clf.fit(Xtr, ytr)
    w, b = clf.coef_[0].astype(np.float64), float(clf.intercept_[0])
    return lambda H: H.astype(np.float64) @ w + b


def fit_mlp(P, N, seed, device, hidden=256, epochs=30, lr=1e-3, wd=1e-4, batch=4096):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    Xtr = torch.as_tensor(np.concatenate([P, N]), dtype=torch.float32)
    ytr = torch.cat([torch.ones(len(P)), torch.zeros(len(N))])
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(len(N) / max(1, len(P)), device=device))
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(Xtr), generator=g)
        for j in range(0, len(Xtr), batch):
            idx = perm[j:j + batch]
            opt.zero_grad()
            lossf(net(Xtr[idx].to(device)).squeeze(1), ytr[idx].to(device)).backward()
            opt.step()
    net.eval()

    def score(H):
        with torch.no_grad():
            return net(torch.as_tensor(H, dtype=torch.float32, device=device)).squeeze(1) \
                .cpu().numpy().astype(np.float64)
    return score


# =====================================================================================
# One (k, seed)
# =====================================================================================

def topk_mean(s, k):
    kk = max(1, min(int(np.ceil(k * len(s))) if 0 < k < 1 else int(k), len(s)))
    return float(np.partition(s, len(s) - kk)[len(s) - kk:].mean())


def run_one(k, seed, X, Y, wsis, gt_idx, protos, args, device):
    import torch
    import torch.nn.functional as F
    from utils import tumor_ratio_metrics as TRM
    from utils.text_topk_mil_torch import make_text_topk

    tr, va = TRM.fewshot_split_ids(Y, k, seed)
    rng = np.random.default_rng(10_000 * k + seed)
    scorers, info = {}, {"k": k, "seed": seed}
    concepts = getattr(args, "concepts", None)

    def new_model():
        return make_text_topk(protos, concepts=concepts, k=args.topk, lam=args.lam,
                              tau=getattr(args, "tau", 1.0), lam_leaf=getattr(args, "lam_leaf", None)).to(device)

    ref = new_model()

    if "zero-shot" in args.scorers:
        def zs(H):
            return ref.zero_shot_scores(torch.as_tensor(H, device=device)).cpu().numpy().astype(np.float64)
        scorers["zero-shot"] = zs
    if "text-topk" in args.scorers:
        m = new_model()
        extra = ({"neg_weight": args.neg_weight, "neg_m": args.neg_m, "neg_margin": args.neg_margin}
                 if args.neg_weight > 0 else {})
        m.fit([X[i] for i in tr], Y[tr], epochs=args.epochs, lr=args.lr, lr_b=args.lr_b, **extra)

        def tt(H, m=m):
            with torch.no_grad():
                return m.patch_scores(torch.as_tensor(H, device=device)).cpu().numpy().astype(np.float64)
        scorers["text-topk"] = tt
        info.update({f"tt_{a}": v for a, v in m.summary().items()})
    if "wr-pseudolabel" in args.scorers:
        from utils.witness_pseudolabel import fit_witness_pseudolabel
        fit_kw = {"epochs": args.epochs, "lr": args.lr, "lr_b": args.lr_b}
        if args.neg_weight > 0:
            fit_kw.update({"neg_weight": args.neg_weight, "neg_m": args.neg_m, "neg_margin": args.neg_margin})
        pos_of = {j: int(i) for j, i in enumerate(tr)}
        gt_sup = {j: gt_idx[i]["gt"] for j, i in pos_of.items() if i in gt_idx}
        pm, diag = fit_witness_pseudolabel([X[i] for i in tr], Y[tr], new_model,
                                           rounds=getattr(args, "pl_rounds", 3), fit_kw=fit_kw,
                                           patch_weight=getattr(args, "pl_patch_weight", 1.0),
                                           z=getattr(args, "pl_z", 3.0), seed=seed, gt=gt_sup)

        def pl(H, m=pm):
            with torch.no_grad():
                return m.patch_scores(torch.as_tensor(H, device=device)).cpu().numpy().astype(np.float64)
        scorers["wr-pseudolabel"] = pl
        d = pd.DataFrame(diag)
        last = d[d["round"] == d["round"].max()]
        info.update({"pl_n_hat_median": float(last.n_hat.median())})
        if "precision" in last:
            info.update({"pl_precision": float(last.precision.mean()),
                         "pl_precision_top3": float(last.precision_top3.mean()),
                         "pl_recall": float(last.recall.mean()),
                         "pl_spearman_n": float(last.n_hat.corr(last.n_true, method="spearman"))})
    if {"oracle-linear", "oracle-mlp"} & set(args.scorers):
        P, N = patch_training_set(tr, Y, X, {i: g["gt"] for i, g in gt_idx.items()}, args.max_neg, rng)
        info.update({"oracle_n_pos": len(P), "oracle_n_neg": len(N)})
        if len(P) == 0:
            print(f"[k={k} seed={seed}] no annotated tumor patches in the support: oracles skipped")
        else:
            if "oracle-linear" in args.scorers:
                scorers["oracle-linear"] = fit_linear(P, N, args.C, seed)
            if "oracle-mlp" in args.scorers:
                scorers["oracle-mlp"] = fit_mlp(P, N, seed, device, epochs=args.mlp_epochs)

    rows = []
    for name, f in scorers.items():
        scores = {i: f(X[i]) for i in va if Y[i] == 0 or i in gt_idx}
        normal = [scores[i] for i in scores if Y[i] == 0]
        pool = np.sort(np.concatenate(normal))
        nmax = np.sort([s.max() for s in normal])
        S = {i: topk_mean(s, args.pool_k) for i, s in scores.items()}
        Sn = np.sort([S[i] for i in S if Y[i] == 0])
        thr_slide = Sn[min(len(Sn) - 1, int(np.ceil(args.spec * len(Sn))) - 1)]
        thr_patch = pool[min(len(pool) - 1, int(np.ceil((1 - args.fpr) * len(pool))))]
        for i, s in scores.items():
            base = {"k": k, "seed": seed, "scorer": name, "WSI": wsis[i], "gt_label": CLASSES[Y[i]],
                    "S": S[i], "detected": float(S[i] > thr_slide)}
            if Y[i] == 0:
                rows.append(base)
                continue
            g = gt_idx[i]["gt"]
            st, sn = s[g], s[~g]
            rows.append({**base, "tumor_area_pct": gt_idx[i]["tumor_area_pct"],
                         "lesion_stage": gt_idx[i]["lesion_stage"],
                         "slide_auc": float(frac_below(Sn, S[i])[0]),
                         "best_beats_normal_max": float(frac_below(nmax, st.max())[0]) if len(st) else np.nan,
                         "tpr_at_fpr": float((st > thr_patch).mean()) if len(st) else np.nan,
                         "auc_within": float(frac_below(np.sort(sn), st).mean()) if len(st) and len(sn) else np.nan})
    return rows, info


# =====================================================================================
# Aggregation, figure, report
# =====================================================================================

def aggregate(df, by, order):
    t = df[df.gt_label == "Tumor"]
    out = []
    for (k, sc), g in t.groupby(["k", "scorer"]):
        for key in order:
            gg = g[g[by] == key]
            ps = gg.groupby("seed")[[m for m, _ in METRICS]].mean() if len(gg) else None
            n = gg.groupby("seed").size().mean() if len(gg) else 0
            r = {"k": k, "scorer": sc, by: key, "n_slides_per_seed": n}
            for m, _ in METRICS:
                r[f"{m}_mean"] = ps[m].mean() if ps is not None else np.nan
                r[f"{m}_sd"] = ps[m].std() if ps is not None else np.nan
            out.append(r)
    return pd.DataFrame(out)


def per_seed(df):
    rows = []
    for (k, sc, seed), g in df.groupby(["k", "scorer", "seed"]):
        t, n = g[g.gt_label == "Tumor"], g[g.gt_label == "Normal"]
        rows.append({"k": k, "scorer": sc, "seed": seed, "slide_auc": t.slide_auc.mean(),
                     "sensitivity_at_spec": t.detected.mean(),
                     "specificity": 1 - n.detected.mean(),
                     "best_beats_normal_max": t.best_beats_normal_max.mean()})
    return pd.DataFrame(rows)


def make_figure(tab, key, order, ks, scorers, xlabel, title, footer, spec, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    x = np.arange(len(order))
    fig, axes = plt.subplots(len(PLOT_METRICS), len(ks), squeeze=False,
                             figsize=(4.6 * len(ks) + 0.6, 3.3 * len(PLOT_METRICS) + 1.0))
    labels = dict(METRICS)
    for r, m in enumerate(PLOT_METRICS):
        for c, k in enumerate(ks):
            ax = axes[r, c]
            for sc in scorers:
                d = tab[(tab.k == k) & (tab.scorer == sc)].set_index(key).reindex(order)
                if d.empty:
                    continue
                y = d[f"{m}_mean"].to_numpy(float)
                e = np.nan_to_num(d[f"{m}_sd"].to_numpy(float))
                ok = np.isfinite(y)
                ax.fill_between(x[ok], (y - e)[ok], np.minimum(1, (y + e)[ok]), color=COLORS[sc],
                                alpha=0.10, lw=0)
                ax.plot(x[ok], y[ok], STYLES[sc], marker=MARKERS[sc], color=COLORS[sc], lw=2.0, ms=6,
                        mec="white", mew=0.8, label=sc)
            n = tab[(tab.k == k)].groupby(key).n_slides_per_seed.max().reindex(order).fillna(0)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{g}\n(n≈{v:.0f})" for g, v in zip(order, n)], fontsize=8)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            lab = labels[m] + (f" {spec:g}" if m == "detected" else "")
            ax.set_title(f"k={k} — {lab}", loc="left", fontsize=9.5, color=INK)
            if r == len(PLOT_METRICS) - 1:
                ax.set_xlabel(xlabel)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 0.965), title="patch scorer (dashed: oracle with patch labels)",
               title_fontsize=8.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "\n".join(textwrap.wrap(footer, 170)), fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


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
    idx = {w: i for i, w in enumerate(wsis)}
    gt_idx = {idx[w]: g for w, g in gt.items()}
    protos = None if getattr(args, "concepts", None) else np.load(args.text_prototypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, infos = [], []
    for k in args.k_shots:
        for seed in tqdm(args.seeds, desc=f"k={k}"):
            r, info = run_one(k, seed, X, Y, wsis, gt_idx, protos, args, device)
            rows += r
            infos.append(info)
    return pd.DataFrame(rows), pd.DataFrame(infos), skipped


def report(df, infos, skipped, args):
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    t = df.gt_label == "Tumor"
    df.loc[t, "wr_bin"] = pd.cut(df.loc[t, "tumor_area_pct"], WR_BINS, labels=WR_LABELS,
                                 include_lowest=True).astype(str)
    df.to_csv(f"{args.out_prefix}_per_slide.csv", index=False)
    ps = per_seed(df).merge(infos, on=["k", "seed"], how="left")
    ps.to_csv(f"{args.out_prefix}_per_seed.csv", index=False)
    by_bin, by_stage = aggregate(df, "wr_bin", WR_LABELS), aggregate(df, "lesion_stage", STAGES)
    by_bin.to_csv(f"{args.out_prefix}_by_bin.csv", index=False)
    by_stage.to_csv(f"{args.out_prefix}_by_stage.csv", index=False)

    scorers = [s for s in SCORERS if s in set(df.scorer)]
    ks = sorted(df.k.unique())
    tag = (f"{len(args.seeds)} seeds, slide score = top-{args.pool_k:g} mean; Text-TopK K={args.topk:g}, "
           f"λ={args.lam:g}" + (f", tail-rank {args.neg_weight:g}" if args.neg_weight > 0 else ""))
    footer = (f"Eval slides only (support excluded). Oracles use annotated patch labels of the k-shot support "
              f"slides (logistic regression C={args.C:g}; MLP 1 hidden layer) and are ceilings, not methods. "
              f"Normal reference = all normal eval slides; detection threshold = {args.spec:g} quantile of their "
              f"slide scores (comparison point). GT: patch centre in annotated tumor.")
    make_figure(by_bin, "wr_bin", WR_LABELS, ks, scorers, "tumor area, % of tissue",
                f"Scorer ceiling by witness rate ({tag})", footer, args.spec,
                f"{args.out_prefix}_by_witness_rate.png")
    make_figure(by_stage, "lesion_stage", STAGES, ks, scorers,
                "largest lesion (ITC < 0.2 mm ≤ micro < 2 mm ≤ macro)",
                f"Scorer ceiling by lesion stage ({tag})", footer, args.spec,
                f"{args.out_prefix}_by_stage.png")

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    o = ps.groupby(["k", "scorer"])[["slide_auc", "sensitivity_at_spec", "specificity",
                                     "best_beats_normal_max"]].mean()
    print("overall (mean over seeds):")
    print(o.to_string(float_format=lambda v: f"{v:.3f}"))
    pl_cols = [c for c in ("pl_n_hat_median", "pl_precision", "pl_precision_top3", "pl_recall", "pl_spearman_n")
               if c in infos]
    if pl_cols:
        print("\nwr-pseudolabel support diagnostics (last round, mean over seeds; precision vs annotated patches):")
        print(infos.groupby("k")[pl_cols].mean().to_string(float_format=lambda v: f"{v:.3f}"))
    for name, tab, key, order in (("tumor area", by_bin, "wr_bin", WR_LABELS),
                                  ("lesion stage", by_stage, "lesion_stage", STAGES)):
        for m in ("best_beats_normal_max", "slide_auc"):
            print(f"\n-- {m} by {name} (columns: scorer)")
            for k in ks:
                pv = tab[tab.k == k].pivot(index=key, columns="scorer", values=f"{m}_mean") \
                    .reindex(index=order, columns=scorers)
                print(f"k={k}")
                print(pv.to_string(float_format=lambda v: f"{v:.3f}"))
    if skipped:
        print(f"\ntumor slides without h5/xml (no GT; excluded from eval and oracle training): {len(skipped)}")
    print(f"\n-> {args.out_prefix}_by_witness_rate.png, _by_stage.png, _per_slide.csv, _by_bin.csv, "
          f"_by_stage.csv, _per_seed.csv")
    return ps, by_bin


def self_test(out_dir):
    from analyze_text_topk_vs_witness_rate import make_synthetic_cohort   # self-test only
    root = make_synthetic_cohort(seed=11, n_slides=80, radii=(500, 800, 1500, 3000, 6000), n_mimic=6)
    args = argparse.Namespace(
        folder=os.path.join(root, "feat"), project="CAMELYON16", encoder="CONCH",
        csv_path=os.path.join(root, "C.csv"), h5_dir=os.path.join(root, "h5"),
        xml_dir=os.path.join(root, "xml"), text_prototypes=os.path.join(root, "protos.npy"),
        k_shots=[4, 8], seeds=[0, 1], scorers=SCORERS, topk=1, lam=0.1, neg_weight=0.0, neg_m=32,
        neg_margin=10.0, epochs=150, lr=5e-3, lr_b=5e-2, C=1.0, max_neg=20000, mlp_epochs=30,
        pool_k=1, fpr=1e-3, spec=0.9, mpp_level0=None, out_prefix=os.path.join(out_dir, "ceiling_self_test"))
    df, infos, skipped = run(args)
    assert not skipped and set(df.scorer) == set(SCORERS) and sorted(df.k.unique()) == [4, 8]
    ps, by_bin = report(df, infos, skipped, args)
    # every scorer is evaluated on the same slides
    assert df.groupby(["k", "seed", "scorer"]).size().unstack().nunique(axis=1).eq(1).all()
    # support slides never evaluated
    from utils import tumor_ratio_metrics as TRM
    meta = pd.read_csv(args.csv_path)
    Y = np.array([CLASSES.index(g) for g in meta.GT])
    tr, _ = TRM.fewshot_split_ids(Y, 8, 1)
    assert not set(meta.WSI.to_numpy()[tr]) & set(df[(df.k == 8) & (df.seed == 1)].WSI)
    # oracle training set: tumor patches of support tumor slides only, negatives capped
    assert (infos.oracle_n_neg <= 20000).all() and (infos.oracle_n_pos > 0).all()
    # the ladder: patch-label oracles must beat zero-shot in the tail on this cohort
    t = df[df.gt_label == "Tumor"].groupby("scorer").best_beats_normal_max.mean()
    print("  mean best_beats_normal_max: " + ", ".join(f"{s} {t[s]:.2f}" for s in SCORERS))
    assert t["oracle-linear"] > t["zero-shot"] + 0.1, t
    # hand check of the training-set builder
    Xs = [np.eye(3, dtype=np.float32)[[0, 1, 2]], np.eye(3, dtype=np.float32)[[2, 2, 0]]]
    P, N = patch_training_set([0, 1], np.array([1, 0]), Xs, {0: np.array([True, False, False])}, 0,
                              np.random.default_rng(0))
    assert P.shape == (1, 3) and N.shape == (5, 3) and P[0, 0] == 1
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
    p.add_argument("--k_shots", type=int, nargs="+", default=[16])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--scorers", nargs="+", choices=SCORERS, default=SCORERS)
    p.add_argument("--topk", type=float, default=3, help="Text-TopK training K.")
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--neg_weight", type=float, default=0.0, help="Text-TopK tail-rank weight (0 = off).")
    p.add_argument("--neg_m", type=int, default=32)
    p.add_argument("--neg_margin", type=float, default=10.0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_b", type=float, default=5e-2)
    p.add_argument("--concepts", default=None,
                   help="Concept file (encode_concepts.py): zero-shot and text-topk rungs use the multi-concept scorer.")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lam_leaf", type=float, default=None)
    p.add_argument("--pl_rounds", type=int, default=3, help="wr-pseudolabel: EM rounds.")
    p.add_argument("--pl_patch_weight", type=float, default=1.0, help="wr-pseudolabel: patch-loss weight.")
    p.add_argument("--pl_z", type=float, default=3.0, help="wr-pseudolabel: excess-count margin (binomial SDs).")
    p.add_argument("--C", type=float, default=1.0, help="Oracle logistic regression inverse L2 strength.")
    p.add_argument("--max_neg", type=int, default=200000,
                   help="Random cap on oracle negative patches (0 = all).")
    p.add_argument("--mlp_epochs", type=int, default=30)
    p.add_argument("--pool_k", type=float, default=1, help="Slide score = top-pool_k mean, all scorers.")
    p.add_argument("--fpr", type=float, default=1e-3)
    p.add_argument("--spec", type=float, default=0.95)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--out_prefix", default="analysis/analysis_ceiling/ceiling")
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(os.path.dirname(os.path.abspath(args.out_prefix)))
        return
    if not args.folder or not args.xml_dir:
        p.error("--folder and --xml_dir are required (or use --self_test)")
    report(*run(args), args)


if __name__ == "__main__":
    main()
