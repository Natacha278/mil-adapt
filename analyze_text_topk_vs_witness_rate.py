"""
analyze_text_topk_vs_witness_rate.py

Performance of Text-TopK MIL (utils/text_topk_mil_torch.py) as a function of witness
rate, over several few-shot seeds: slide-level accuracy, and the patch-level MECHANISM
the method is meant to fix, measured for the learned scorer and for the zero-shot text
direction it starts from (Delta = 0) on exactly the same patches.

For each seed the model is trained on that seed's support slides (same k-per-class
draw as main.py / MIL-Adapter) and every eval slide is scored with the full features.

SLIDE LEVEL (decision rule --threshold)
  zero        Tumor iff S > 0 (steps 1-3)
  conformal   step 4: leave-one-normal-out conformal threshold at target false-positive
              rate --alpha (utils/conformal.py); slide score = margin (mean S_-j(x) - z_j).
              The specificity panel then shows the 1 - nominal target to check against.
  detection rate   tumor slides predicted Tumor, per witness-rate bin / stage
  specificity      normal slides predicted Normal
  slide AUC        tumor slides of the bin vs all normal eval slides (threshold-free)
  Patch metrics always use the model trained on the full support (same for both rules).

PATCH LEVEL, per tumor slide, for scorer in {learned, zero-shot}
  best_beats_normal_max  share of normal eval slides whose HIGHEST patch scores below this
                         slide's best tumor patch. The quantity a max/top-K rule depends
                         on; exact here (every patch of every normal slide is scored).
  tpr_at_fpr             tumor patches above the (1 - --fpr) quantile of all normal-slide
                         patches (default fpr 1e-3)
  top1_in_tumor          the slide's highest-scoring patch is a tumor patch (0/1)
  topk_precision         share of the slide's top-K patches that are tumor patches
  auc_within             tumor vs the same slide's normal patches
  auc_vs_normal_slides   tumor patches vs all normal-slide patches
  Patch scores are rank-compared within one scorer, so the learned bias cancels.

GROUND TRUTH and WITNESS RATE: utils/witness_rate.py (true patch footprint, "_2" holes
removed); x-axis = tumor area, % of tissue; stage = largest lesion.

AGGREGATION per seed: rates are means over the bin's slides, other patch metrics are
medians; then mean and SD across seeds.

OUTPUTS <out_prefix>_per_slide.csv, _per_seed.csv, _by_bin.csv, _by_stage.csv,
        <out_prefix>_by_witness_rate.png, <out_prefix>_by_stage.png

    python analyze_text_topk_vs_witness_rate.py --self_test
    python analyze_text_topk_vs_witness_rate.py \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
        --xml_dir $SCRATCH/camelyon16/annotations \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --topk 3 --lam 0.1 --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out_prefix analysis/analysis_text_topk/mechanism_K1_lam0.1
    # step 4: same, plus  --threshold conformal --alpha 0.1  (and a _conf out_prefix)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CLASSES = ["Normal", "Tumor"]
WR_BINS = [0.0, 0.1, 0.5, 2.0, 10.0, 100.0]
WR_LABELS = [f"{lo:g}–{hi:g}%" for lo, hi in zip(WR_BINS[:-1], WR_BINS[1:])]
STAGES = ["ITC", "micro", "macro"]
SCORERS = ["learned", "zero-shot"]
PATCH_METRICS = ["best_beats_normal_max", "tpr_at_fpr", "top1_in_tumor", "topk_precision",
                 "auc_within", "auc_vs_normal_slides"]
RATE_METRICS = {"top1_in_tumor", "slide_correct"}          # averaged, not median-ed


# =====================================================================================
# Metrics (numpy)
# =====================================================================================

def frac_below(sorted_ref, values):
    """For each value: share of sorted_ref strictly below it, ties counted 1/2."""
    v = np.atleast_1d(np.asarray(values, float))
    lt = np.searchsorted(sorted_ref, v, "left")
    le = np.searchsorted(sorted_ref, v, "right")
    return (lt + 0.5 * (le - lt)) / len(sorted_ref)


def patch_metrics(s, gt, k, normal_pool, normal_max, fpr):
    """s: patch scores of one tumor slide (one scorer); gt: bool tumor mask."""
    if gt.sum() == 0:
        return {m: np.nan for m in PATCH_METRICS}
    st, sn = s[gt], s[~gt]
    kk = max(1, min(int(np.ceil(k * len(s))) if 0 < k < 1 else int(k), len(s)))
    top = np.argpartition(-s, kk - 1)[:kk]
    thr = normal_pool[min(len(normal_pool) - 1, int(np.ceil((1 - fpr) * len(normal_pool))))]
    return {"best_beats_normal_max": float(frac_below(normal_max, st.max())[0]),
            "tpr_at_fpr": float((st > thr).mean()),
            "top1_in_tumor": float(gt[int(np.argmax(s))]),
            "topk_precision": float(gt[top].mean()),
            "auc_within": float(frac_below(np.sort(sn), st).mean()) if len(sn) else np.nan,
            "auc_vs_normal_slides": float(frac_below(normal_pool, st).mean())}


def seed_rows(seed, scored, gt_cache, labels, k, fpr):
    """scored: {wsi: {"S": slide score, "learned": patch scores, "zero-shot": patch scores}}"""
    pools, maxima = {}, {}
    for sc in SCORERS:
        norm = [scored[w][sc] for w in scored if labels[w] == "Normal"]
        pools[sc] = np.sort(np.concatenate(norm)) if norm else np.array([np.inf])
        maxima[sc] = np.sort([v.max() for v in norm]) if norm else np.array([np.inf])
    S_norm = np.sort([scored[w]["S"] for w in scored if labels[w] == "Normal"])
    rows = []
    for w, sc in scored.items():
        pred = CLASSES[int(sc.get("pred", sc["S"] > 0))]
        base = {"seed": seed, "WSI": w, "gt_label": labels[w], "S": sc["S"], "pred": pred,
                "slide_correct": float(pred == labels[w])}
        if "p_value" in sc:
            base["p_value"] = sc["p_value"]
        if labels[w] == "Normal":
            rows.append({**base, "scorer": "slide"})
            continue
        g = gt_cache[w]
        info = {kk: g[kk] for kk in ("tumor_area_pct", "witness_rate_centre_pct",
                                     "lesion_stage", "max_lesion_diam_mm", "n_tumor_patches")}
        rows.append({**base, **info, "scorer": "slide",
                     "slide_auc_placement": float(frac_below(S_norm, sc["S"])[0])})
        for s_name in SCORERS:
            rows.append({**base, **info, "scorer": s_name,
                         **patch_metrics(sc[s_name], g["gt"], k, pools[s_name],
                                         maxima[s_name], fpr)})
    return rows


def aggregate(df, by, order):
    t = df[df.gt_label == "Tumor"]
    out = []
    slide = t[t.scorer == "slide"]
    for key in order:
        g = slide[slide[by] == key]
        ps = g.groupby("seed").agg(det=("slide_correct", "mean"),
                                   auc=("slide_auc_placement", "mean"), n=("WSI", "size"))
        out.append({by: key, "scorer": "slide", "n_slides_per_seed": ps.n.mean() if len(ps) else 0,
                    "detection_rate_mean": ps.det.mean(), "detection_rate_sd": ps.det.std(),
                    "slide_auc_mean": ps.auc.mean(), "slide_auc_sd": ps.auc.std()})
        for s_name in SCORERS:
            g = t[(t.scorer == s_name) & (t[by] == key)]
            agg = {m: (m, "mean" if m in RATE_METRICS else "median") for m in PATCH_METRICS}
            ps = g.groupby("seed").agg(**agg) if len(g) else pd.DataFrame(columns=PATCH_METRICS)
            r = {by: key, "scorer": s_name}
            for m in PATCH_METRICS:
                r[f"{m}_mean"] = ps[m].mean() if len(ps) else np.nan
                r[f"{m}_sd"] = ps[m].std() if len(ps) else np.nan
            out.append(r)
    return pd.DataFrame(out)


def per_seed_overall(df):
    s = df[df.scorer == "slide"]
    rows = []
    for seed, g in s.groupby("seed"):
        t, n = g[g.gt_label == "Tumor"], g[g.gt_label == "Normal"]
        sens, spec = t.slide_correct.mean(), n.slide_correct.mean()
        rows.append({"seed": seed, "accuracy": g.slide_correct.mean(), "sensitivity": sens,
                     "specificity": spec, "balanced_accuracy": (sens + spec) / 2,
                     "slide_auc": t.slide_auc_placement.mean()})
    return pd.DataFrame(rows)


# =====================================================================================
# Figures
# =====================================================================================

C_LEARN, C_ZS, C_SLIDE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"


def make_figure(tab, groups, per_seed, xlabel, title, footer, out_png, target_spec=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    x = np.arange(len(groups))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))

    def band_line(ax, y, e, color, label, ls="-", marker="o"):
        y, e = np.asarray(y, float), np.nan_to_num(np.asarray(e, float))
        ok = np.isfinite(y)
        ax.fill_between(x[ok], (y - e)[ok], np.minimum(1, (y + e)[ok]), color=color, alpha=0.12, lw=0)
        ax.plot(x[ok], y[ok], ls, marker=marker, color=color, lw=2.1, ms=5.5, mec="white", mew=0.8,
                label=label)

    sl = tab[tab.scorer == "slide"].set_index(tab.columns[0]).reindex(groups)
    a = axes[0]
    band_line(a, sl.detection_rate_mean, sl.detection_rate_sd, C_SLIDE,
              "tumor slides detected (" + ("S > 0" if target_spec is None else "p ≤ α") + ")")
    band_line(a, sl.slide_auc_mean, sl.slide_auc_sd, INK2, "slide AUC vs normal slides", ls="--", marker="s")
    sp, sps = per_seed.specificity.mean(), per_seed.specificity.std()
    a.axhspan(sp - sps, min(1, sp + sps), color=C_ZS, alpha=0.10, lw=0)
    a.axhline(sp, color=C_ZS, lw=1.4, ls=":", label=f"normal slides kept Normal ({sp:.2f})")
    if target_spec is not None:
        a.axhline(target_spec, color=INK, lw=1.0, ls=(0, (1, 3)),
                  label=f"conformal target specificity ({target_spec:.2f})")
    a.set_title("Slide level (Text-TopK)", loc="left", fontsize=10, color=INK)

    for ax, metrics, ttl in [
        (axes[1], [("best_beats_normal_max", "-", "o"), ("tpr_at_fpr", "--", "s")],
         "Tail: lesion vs normal slides' extreme patches"),
        (axes[2], [("top1_in_tumor", "-", "o"), ("auc_within", "--", "s")],
         "Selection: is the top patch the lesion?"),
    ]:
        for s_name, col in (("learned", C_LEARN), ("zero-shot", C_ZS)):
            d = tab[tab.scorer == s_name].set_index(tab.columns[0]).reindex(groups)
            for m, ls, mk in metrics:
                band_line(ax, d[f"{m}_mean"], d[f"{m}_sd"], col,
                          f"{s_name}: {m.replace('_', ' ')}", ls=ls, marker=mk)
        ax.set_title(ttl, loc="left", fontsize=10, color=INK)

    n = sl.n_slides_per_seed
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n≈{n.get(g, 0):.0f})" for g in groups])
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=7.6, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                  ncol=2, columnspacing=1.2, handlelength=2.6)
    axes[0].set_ylabel("rate / AUC (mean ± SD over seeds)")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, footer, fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================================================
# Pipeline
# =====================================================================================

def ground_truth(wsis, labels, X, h5_dir, xml_dir, gt_rule, min_occ, mpp):
    from utils import witness_rate as W
    cache, skipped = {}, []
    for i, w in enumerate(wsis):
        if labels[w] != "Tumor":
            continue
        h5, xml = os.path.join(h5_dir, f"{w}.h5"), os.path.join(xml_dir, f"{w}.xml")
        if not (os.path.exists(h5) and os.path.exists(xml)):
            skipped.append(w)
            continue
        grid = W.load_patch_grid(h5)
        if len(grid) != len(X[i]):
            raise SystemExit(f"{w}: {len(X[i])} patches in the .npy but {len(grid)} in the .h5")
        ann = W.parse_camelyon16_xml(xml)
        occ = W.patch_occupancy_from_polygons(grid.coords, grid.patch_size, ann)
        cen = W.patch_centre_in_polygons(grid.coords, grid.patch_size, ann)
        st = W.slide_witness_stats(occ, cen)
        geo = W.lesion_geometry(ann, mpp)
        gt = cen if gt_rule == "centre" else occ >= min_occ
        dmax = float(geo.diam_max_mm.max()) if len(geo) else np.nan
        cache[w] = {"gt": gt, "tumor_area_pct": st["tumor_area_pct"],
                    "witness_rate_centre_pct": st["witness_rate_centre_pct"],
                    "max_lesion_diam_mm": dmax, "n_tumor_patches": int(gt.sum()),
                    "lesion_stage": W.stage_of(dmax) if np.isfinite(dmax) else "none"}
    return cache, skipped


def run(args):
    import torch
    from tqdm import tqdm
    from utils import tumor_ratio_metrics as TRM
    from utils.text_topk_mil_torch import ConformalTextTopK, MultiConceptTextTopKMIL, make_text_topk
    from utils.conformal import count_max
    if args.threshold == "conformal":
        count_max(args.k_shots, args.alpha)                  # fail before any training
    from utils import witness_rate as W

    meta = pd.read_csv(args.csv_path)
    wsis = meta.WSI.to_numpy()
    labels = dict(zip(meta.WSI, meta.GT))
    Y = np.array([CLASSES.index(g) for g in meta.GT])
    X = [np.load(os.path.join(args.folder, args.project, args.encoder, f"{w}.npy")).astype(np.float32)
         for w in tqdm(wsis, desc="loading features")]
    gt_cache, skipped = ground_truth(wsis, labels, X, args.h5_dir, args.xml_dir, args.gt,
                                     args.min_occupancy, args.mpp_level0 or W.DEFAULT_MPP_LEVEL0)
    concepts = getattr(args, "concepts", None)
    protos = None if concepts else np.load(args.text_prototypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def new_model():
        return make_text_topk(protos, concepts=concepts, k=args.topk, lam=args.lam,
                              tau=getattr(args, "tau", 1.0), lam_leaf=getattr(args, "lam_leaf", None)).to(device)

    tail_m = getattr(args, "tail_m", 32)
    rows, train_log, tail_rows = [], [], []
    for seed in args.seeds:
        tr, va = TRM.fewshot_split_ids(Y, args.k_shots, seed)
        fit_kw = dict(epochs=args.epochs, lr=args.lr, lr_b=args.lr_b, neg_weight=args.neg_weight,
                      neg_m=args.neg_m, neg_margin=args.neg_margin)
        model = new_model()
        model.fit([X[i] for i in tr], Y[tr], **fit_kw)
        log = {"seed": seed, **model.summary()}
        det = None
        if args.threshold == "conformal":
            det = ConformalTextTopK.fit_loo(
                [X[i] for i in tr], Y[tr],
                new_model,
                alpha=args.alpha, **fit_kw)
            with torch.no_grad():
                ins = np.array([float(model.slide_score(torch.as_tensor(X[i], device=device))[0])
                                for i in tr[Y[tr] == 0]])
            log.update({**{f"conf_{a}": v for a, v in det.summary().items()},
                        "conf_optimism": float(np.mean(det.z.cpu().numpy() - ins))})
        train_log.append(log)
        multi = isinstance(model, MultiConceptTextTopKMIL)
        tail_counts = {"learned": {}, "zero-shot": {}}
        scored = {}
        with torch.no_grad():
            for i in tqdm(va, desc=f"seed {seed}", leave=False):
                w = wsis[i]
                if labels[w] == "Tumor" and w not in gt_cache:
                    continue
                H = torch.as_tensor(X[i], device=device)
                learned = model.patch_scores(H)
                kk = model._k(len(learned))
                S = float(torch.topk(learned, kk).values.mean())
                zs = model.zero_shot_scores(H)
                if multi and labels[w] == "Normal":
                    # which normal concept explains this normal slide's most tumor-like patches
                    for name, sc, z in (("learned", learned, False), ("zero-shot", zs, True)):
                        top = torch.topk(sc, min(tail_m, len(sc))).indices
                        _, nwin = model.explain(H[top], zero_shot=z)
                        for c in nwin.tolist():
                            tail_counts[name][c] = tail_counts[name].get(c, 0) + 1
                scored[w] = {"S": S, "learned": learned.cpu().numpy().astype(np.float64),
                             "zero-shot": zs.cpu().numpy().astype(np.float64)}
                if det is not None:
                    e = det.evaluate(H)
                    scored[w].update({"S": e["margin"], "pred": e["pred"],
                                      "p_value": e["p_value"]})
        rows += seed_rows(seed, scored, gt_cache, labels, args.topk, args.fpr)
        if multi:
            for name, cnt in tail_counts.items():
                tot = max(1, sum(cnt.values()))
                for c in torch.where(~model.tumor_mask)[0].tolist():
                    tail_rows.append({"seed": seed, "scorer": name, "concept": model.names[c],
                                      "n": cnt.get(c, 0), "share": cnt.get(c, 0) / tot})
        print(f"[seed {seed}] {model.summary()}")
    df = pd.DataFrame(rows)
    if tail_rows:
        df.attrs["tail_composition"] = pd.DataFrame(tail_rows)
    return df, pd.DataFrame(train_log), skipped


def report(df, train_log, skipped, args):
    t = df.gt_label == "Tumor"
    df.loc[t, "wr_bin"] = pd.cut(df.loc[t, "tumor_area_pct"], WR_BINS, labels=WR_LABELS,
                                 include_lowest=True).astype(str)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    df.to_csv(f"{args.out_prefix}_per_slide.csv", index=False)
    ps = per_seed_overall(df)
    ps = ps.merge(train_log, on="seed", how="left") if len(train_log) else ps
    ps.to_csv(f"{args.out_prefix}_per_seed.csv", index=False)
    by_bin = aggregate(df, "wr_bin", WR_LABELS)
    by_stage = aggregate(df, "lesion_stage", STAGES)
    by_bin.to_csv(f"{args.out_prefix}_by_bin.csv", index=False)
    by_stage.to_csv(f"{args.out_prefix}_by_stage.csv", index=False)

    rule = (f"conformal α={args.alpha:g}" if args.threshold == "conformal" else "S > 0")
    if getattr(args, "concepts", None):
        rule += f", multi-concept τ={getattr(args, 'tau', 1.0):g}"
    if getattr(args, "neg_weight", 0) > 0:
        rule += f", tail-rank {args.neg_weight:g} (M={args.neg_m}, margin {args.neg_margin:g})"
    title_tag = (f"Text-TopK (K={args.topk:g}, λ={args.lam:g}, {rule}), k={args.k_shots}, "
                 f"{len(args.seeds)} seeds")
    target = (1 - ps.conf_nominal_fpr.mean()) if "conf_nominal_fpr" in ps else None
    acc = (f"accuracy {ps.accuracy.mean():.3f} ± {ps.accuracy.std():.3f}, balanced "
           f"{ps.balanced_accuracy.mean():.3f}, slide AUC {ps.slide_auc.mean():.3f}")
    footer = (f"Eval slides only (support slides excluded). Patch metrics: learned scorer vs the "
              f"zero-shot text direction on the same patches; normal reference = every patch of the "
              f"seed's normal eval slides; TPR at FPR {args.fpr:g}. GT: patch "
              f"{'centre' if args.gt == 'centre' else f'occupancy >= {args.min_occupancy:g}'} in tumor.")
    make_figure(by_bin, WR_LABELS, ps, "tumor area, % of tissue",
                f"{title_tag} by witness rate: {acc}", footer, f"{args.out_prefix}_by_witness_rate.png",
                target)
    make_figure(by_stage, STAGES, ps, "largest lesion (ITC < 0.2 mm ≤ micro < 2 mm ≤ macro)",
                f"{title_tag} by lesion stage: {acc}", footer, f"{args.out_prefix}_by_stage.png",
                target)

    print("\n" + "=" * 104)
    print(f"{title_tag}: {acc}; specificity {ps.specificity.mean():.3f}"
          + (f" (conformal target {target:.3f}; in-sample optimism of support normals "
             f"{ps.conf_optimism.mean():+.3f})" if target is not None else ""))
    print("=" * 104)
    for name, tab, key in (("witness rate", by_bin, "wr_bin"), ("lesion stage", by_stage, "lesion_stage")):
        sl = tab[tab.scorer == "slide"].set_index(key)[["n_slides_per_seed", "detection_rate_mean",
                                                         "slide_auc_mean"]]
        cols = ["best_beats_normal_max_mean", "tpr_at_fpr_mean", "top1_in_tumor_mean"]
        piv = pd.concat([tab[tab.scorer == s].set_index(key)[cols].rename(
            columns=lambda c, s=s: f"{s}: {c.replace('_mean', '')}") for s in SCORERS], axis=1)
        print(f"by {name}:")
        print(pd.concat([sl, piv], axis=1).to_string(float_format=lambda v: f"{v:.3f}"))
        print("-" * 104)
    tail = df.attrs.get("tail_composition")
    if tail is not None and len(tail):
        tail.to_csv(f"{args.out_prefix}_tail_composition.csv", index=False)
        tab = tail.groupby(["concept", "scorer"]).share.mean().unstack("scorer")
        print(f"normal-tail composition (top-{getattr(args, 'tail_m', 32)} patches of each normal eval slide, "
              f"winning normal concept; mean share over seeds):")
        print(tab.to_string(float_format=lambda v: f"{v:.3f}"))
        print("-" * 104)
    if skipped:
        print(f"tumor slides without h5/xml, skipped: {len(skipped)} ({skipped[:5]})")
    print(f"-> {args.out_prefix}_by_witness_rate.png, _by_stage.png, _per_slide.csv, _per_seed.csv, "
          f"_by_bin.csv, _by_stage.csv")


# =====================================================================================
# Self-test: a small synthetic cohort on disk (npy + h5 + XML), run end to end
# =====================================================================================

def make_synthetic_cohort(seed=3, n_slides=72, radii=(600, 1500, 3000, 6000), side=24,
                          n_mimic=4, d=64, root=None):
    """Synthetic CAMELYON16-like cohort on disk: npy features (L2-normalised), TRIDENT-style
    h5 coords (1024 px patches) and ASAP XML lesions; half normal, half tumor, one circular
    lesion of radius radii[i % len(radii)] (level-0 px). Every slide has n_mimic look-alike
    patches (tumor-like along the text direction w0, opposite along the true tumor axis u).
    Returns the root folder (feat/CAMELYON16/CONCH, h5, xml, protos.npy, C.csv)."""
    import h5py
    import tempfile
    rng = np.random.default_rng(seed)
    root = root or tempfile.mkdtemp()
    ps = 1024
    tT, tN = rng.standard_normal(d), rng.standard_normal(d)
    P = np.column_stack([tN, tT]); P /= np.linalg.norm(P, axis=0)
    w0 = P[:, 1] - P[:, 0]; w0 /= np.linalg.norm(w0)
    u = rng.standard_normal(d); u -= (u @ w0) * w0; u /= np.linalg.norm(u)
    for sub in ("feat/CAMELYON16/CONCH", "h5", "xml"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    np.save(os.path.join(root, "protos.npy"), P)
    gx, gy = np.meshgrid(np.arange(side), np.arange(side))
    coords = np.column_stack([gx.ravel(), gy.ravel()]).astype(float) * ps
    th = np.linspace(0, 2 * np.pi, 40)[:-1]
    rows = []
    for i in range(n_slides):
        w = f"s{i:03d}"; tumor = i >= n_slides // 2
        H = rng.standard_normal((len(coords), d)) * 0.05
        mimic = rng.choice(len(coords), n_mimic, replace=False)
        H[mimic] += 0.24 * w0 - 0.22 * u
        if tumor:
            r = radii[i % len(radii)]
            c = rng.uniform(6 * ps, 18 * ps, 2)
            poly = c + np.column_stack([r * np.cos(th), r * np.sin(th)])
            xml = "<ASAP_Annotations><Annotations><Annotation PartOfGroup='_0'><Coordinates>" + "".join(
                f'<Coordinate Order="{k}" X="{x}" Y="{y}"/>' for k, (x, y) in enumerate(poly)) + \
                "</Coordinates></Annotation></Annotations></ASAP_Annotations>"
            open(os.path.join(root, "xml", f"{w}.xml"), "w").write(xml)
            from matplotlib.path import Path as MP
            inside = MP(poly).contains_points(coords + ps / 2)
            H[inside] += 0.22 * w0 + 0.22 * u
        np.save(os.path.join(root, "feat/CAMELYON16/CONCH", f"{w}.npy"),
                (H / np.linalg.norm(H, axis=1, keepdims=True)).astype(np.float32))
        with h5py.File(os.path.join(root, "h5", f"{w}.h5"), "w") as f:
            f.create_dataset("coords", data=coords).attrs["patch_size_level0"] = ps
        rows.append({"case_id": w, "WSI": w, "GT": "Tumor" if tumor else "Normal"})
    pd.DataFrame(rows).to_csv(os.path.join(root, "C.csv"), index=False)
    return root


def self_test(out_dir):
    root = make_synthetic_cohort()
    args = argparse.Namespace(
        folder=os.path.join(root, "feat"), project="CAMELYON16", encoder="CONCH",
        csv_path=os.path.join(root, "C.csv"), h5_dir=os.path.join(root, "h5"),
        xml_dir=os.path.join(root, "xml"), text_prototypes=os.path.join(root, "protos.npy"),
        k_shots=6, seeds=[0, 1, 2], topk=1, lam=0.1, epochs=150, lr=5e-3, lr_b=5e-2,
        gt="centre", min_occupancy=0.5, fpr=1e-3, mpp_level0=None, threshold="zero", alpha=0.1,
        neg_weight=0.0, neg_m=32, neg_margin=10.0, concepts=None, tau=1.0, lam_leaf=None, tail_m=32,
        out_prefix=os.path.join(out_dir, "text_topk_self_test"))
    df, log, skipped = run(args)
    report(df, log, skipped, args)
    t = df[df.gt_label == "Tumor"]
    lr_ = t[t.scorer == "learned"].best_beats_normal_max.mean()
    zs_ = t[t.scorer == "zero-shot"].best_beats_normal_max.mean()
    assert lr_ > zs_ + 0.1, (lr_, zs_)
    # hand check of one patch-metric row
    pool = np.sort(np.array([0.0, 1.0, 2.0, 3.0])); nmax = np.sort(np.array([1.0, 3.0]))
    m = patch_metrics(np.array([2.5, 0.5, 3.5, 0.2]), np.array([True, False, False, False]),
                      1, pool, nmax, 0.25)
    assert m["best_beats_normal_max"] == 0.5 and m["top1_in_tumor"] == 0.0
    assert m["auc_within"] == 2 / 3 and m["auc_vs_normal_slides"] == 0.75

    # step 4: same cohort with the conformal rule (6 normals -> alpha >= 1/7)
    cargs = argparse.Namespace(**{**vars(args), "threshold": "conformal", "alpha": 0.15,
                                  "out_prefix": args.out_prefix + "_conf"})
    try:
        run(argparse.Namespace(**{**vars(cargs), "alpha": 0.1}))
        raise AssertionError("alpha < 1/(k_shots+1) accepted")
    except ValueError:
        pass
    cdf, clog, cskipped = run(cargs)
    report(cdf, clog, cskipped, cargs)
    key = ["seed", "WSI", "scorer"]
    pm = df[df.scorer != "slide"].set_index(key)[PATCH_METRICS]
    cpm = cdf[cdf.scorer != "slide"].set_index(key)[PATCH_METRICS].loc[pm.index]
    assert np.allclose(pm.values, cpm.values, equal_nan=True)   # patch metrics: full model, both rules
    sl = cdf[cdf.scorer == "slide"]
    assert set(sl.pred) <= set(CLASSES) and sl.p_value.between(1 / 7, 1).all()
    assert ((sl.p_value <= 0.15) == (sl.pred == "Tumor")).all()
    ps0, ps1 = per_seed_overall(df), per_seed_overall(cdf)
    assert "conf_optimism" in clog and (clog.conf_nominal_fpr == 1 / 7).all()
    print(f"  conformal: specificity {ps1.specificity.mean():.3f} (target {6 / 7:.3f}) vs S > 0 "
          f"{ps0.specificity.mean():.3f}; sensitivity {ps1.sensitivity.mean():.3f} vs "
          f"{ps0.sensitivity.mean():.3f}; in-sample optimism {clog.conf_optimism.mean():+.3f}")
    assert ps1.specificity.mean() >= 0.7

    # normal-tail ranking loss: runs end to end and changes the learned scorer
    rargs = argparse.Namespace(**{**vars(args), "neg_weight": 1.0, "out_prefix": args.out_prefix + "_rank"})
    rdf, rlog, rskipped = run(rargs)
    report(rdf, rlog, rskipped, rargs)
    assert not np.allclose(rlog.delta_norm.values, log.delta_norm.values)
    lr_r = rdf[(rdf.gt_label == "Tumor") & (rdf.scorer == "learned")].best_beats_normal_max.mean()
    print(f"  tail-rank loss: best tumor patch beats normal-slide max {lr_r:.2f} (without: {lr_:.2f})")

    # multi-concept scorer: tumor + normal from the cohort's prompts, plus one extra normal concept
    P = np.load(args.text_prototypes)
    rng_c = np.random.default_rng(5)
    extra = rng_c.standard_normal(P.shape[0])
    cpath = os.path.join(os.path.dirname(args.text_prototypes), "concepts.npz")
    np.savez(cpath, protos=np.column_stack([P[:, 1], P[:, 0], extra]).astype(np.float32),
             names=np.array(["tumor", "normal", "other_normal"]), is_tumor=np.array([1, 0, 0], bool))
    margs = argparse.Namespace(**{**vars(args), "concepts": cpath, "tau": 1.0, "lam_leaf": None,
                                  "tail_m": 8, "seeds": [0, 1], "out_prefix": args.out_prefix + "_mc"})
    mdf, mlog, mskipped = run(margs)
    report(mdf, mlog, mskipped, margs)
    tail = mdf.attrs["tail_composition"]
    assert set(tail.concept) == {"normal", "other_normal"} and set(tail.scorer) == {"learned", "zero-shot"}
    assert np.allclose(tail.groupby(["seed", "scorer"]).share.sum(), 1.0)
    assert "n_concepts" in mlog and (mlog.n_concepts == 3).all()
    print("  multi-concept run: tail composition written, shares sum to 1 per seed and scorer")
    print(f"self_test: OK (best tumor patch beats normal-slide max: learned {lr_:.2f} vs zero-shot {zs_:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--csv_path", default="./local_data/csv/CAMELYON16.csv")
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--xml_dir")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--k_shots", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--topk", type=float, default=1)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_b", type=float, default=5e-2)
    p.add_argument("--gt", choices=["centre", "occ"], default="centre")
    p.add_argument("--min_occupancy", type=float, default=0.5)
    p.add_argument("--fpr", type=float, default=1e-3)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--neg_weight", type=float, default=0.0,
                   help="Weight of the normal-tail ranking loss (0 = off).")
    p.add_argument("--neg_m", type=int, default=32)
    p.add_argument("--neg_margin", type=float, default=10.0)
    p.add_argument("--concepts", default=None,
                   help="Concept file from encode_concepts.py: multi-concept scorer (zero-shot rung = "
                        "the same concepts without training).")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lam_leaf", type=float, default=None)
    p.add_argument("--tail_m", type=int, default=32,
                   help="Multi-concept: top patches per normal slide used for the tail composition.")
    p.add_argument("--threshold", choices=["zero", "conformal"], default="zero")
    p.add_argument("--alpha", type=float, default=0.1,
                   help="Conformal: target false-positive rate on normal slides (>= 1/(k_shots+1)).")
    p.add_argument("--out_prefix", default="analysis/analysis_text_topk/mechanism")
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(os.path.dirname(os.path.abspath(args.out_prefix)))
        return
    if not args.folder or not args.xml_dir:
        p.error("--folder and --xml_dir are required (or use --self_test)")
    df, log, skipped = run(args)
    report(df, log, skipped, args)


if __name__ == "__main__":
    main()
