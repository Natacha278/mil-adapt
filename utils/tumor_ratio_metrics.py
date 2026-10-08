"""
utils/tumor_ratio_metrics.py

Shared estimators for "how does discrimination depend on tumor burden", used by BOTH
analyze_tumor_ratio_miladapter.py and analyze_tumor_ratio_mizero.py.

The two entry points exist to compare a mean-like aggregator (MIL-Adapter: every
aggregator in utils/MIL/ builds the bag vector as a convex combination of instances)
against an order statistic (MI-Zero top-K). That comparison is only meaningful if both
arms are measured with literally the same code, which is why the estimators live here
rather than being copied into each script.

WHY NOT ACCURACY PER TUMOR-RATIO BIN
------------------------------------
analyze_tumor_pct_accuracy.py runs with --tumor_only on, so every stratum has
n_normal_gt = 0 and the per-bin number is RECALL at whatever threshold the model
happened to settle on. Discrimination and operating point are not separable from it. On
this evaluation split, one score vector read at two thresholds -- identical
discrimination by construction -- gives sensitivity curves differing by ~0.39 on average
across bins.

Worse, the degenerate case is exact rather than approximate: AUC computed from BINARY
predictions equals (TPR - FPR + 1)/2 = balanced accuracy, and per stratum it equals
(sens_bin + global specificity)/2 -- an affine rescaling of the sensitivity curve, with
no new information. That identity is why both entry points score slides themselves and
keep the continuous value, instead of reading a saved pred_label.

WHAT IS COMPUTED
----------------
    AUC_b = 1/(|T_b|*|N|) * sum_{i in T_b} sum_{j in N} [1(s_i > s_j) + 0.5*1(s_i = s_j)]

the Mann-Whitney statistic of stratum b against the FULL normal pool N -- the same
denominator pool in every stratum, so strata are mutually comparable. Normal slides are
kept out of the BINNING (they all sit at 0% burden by construction) but used in full as
the reference pool.

Plus:
  * sensitivity at a globally fixed specificity (threshold chosen once on the whole
    normal pool, applied unchanged in every stratum);
  * placement values u_i = rank of tumor slide i inside the normal pool, whose mean over
    any subset IS that subset's AUC -- so the headline can be a binning-free regression
    u ~ alpha + gamma*log10(pi), one number with a CI;
  * AUC_low / AUC_high either side of the beta = 1/2 detection wall, and their gap.

VARIANCE: TWO COMPONENTS THAT DO NOT COMBINE
--------------------------------------------
Slide sampling -> paired bootstrap, resampling the normal pool ONCE per replicate and
sharing it across every stratum and seed (strata share negatives, so their AUCs are
positively correlated; independent resampling would give wrong intervals for
DIFFERENCES). Training randomness -> seed spread, reported separately. Averaging over
seeds does not shrink the slide-sampling variance, because every seed is evaluated on
the same slides.

BIN EDGES ARE FIXED, NOT QUANTILES
----------------------------------
analyze_tumor_pct_accuracy.py defaults to pd.qcut, so edges move with whichever slides
landed in the run -- the committed outputs have (0.049, 0.098] / (0.048, 0.1] /
(0.1, 0.21] / (0.1, 0.23] / (0.1, 0.22] for nominally the same bin. Methods then get
compared on bins that are not the same bins.
"""

import numpy as np
import pandas as pd

# Fixed, pre-registered edges in PERCENT of tissue patches, spaced roughly
# logarithmically because tumor burden is log-distributed.
DEFAULT_EDGES = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 100.0]
SPECS = (0.90, 0.95)


# =====================================================================================
# Estimators
# =====================================================================================

def placement_values(pos, neg):
    """u_i = fraction of the negative pool that tumor slide i outranks (ties at 0.5).

    mean(u) over any subset is exactly that subset's AUC against `neg`, which is what
    lets everything downstream run off one per-slide quantity. O((n+m) log(n+m)).
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(neg) == 0:
        return np.full(len(pos), np.nan)
    s = np.sort(neg)
    lt = np.searchsorted(s, pos, side="left")
    le = np.searchsorted(s, pos, side="right")
    return (lt + 0.5 * (le - lt)) / len(neg)


def auc_from_placement(u):
    u = np.asarray(u, float)
    return float(np.nanmean(u)) if len(u) else np.nan


def sens_at_spec(pos, neg, spec):
    """Sensitivity at a threshold fixed once on the WHOLE negative pool."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, np.nan
    thr = float(np.quantile(neg, spec))
    return float((pos > thr).mean()), thr


def slope_u_vs_logpi(u, pi):
    """OLS slope of placement value on log10 burden: AUC change per decade."""
    m = np.isfinite(u) & np.isfinite(pi) & (pi > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    b, a = np.polyfit(np.log10(pi[m]), u[m], 1)
    return float(b), float(a)


def _pava(y):
    """Pool-adjacent-violators: least-squares fit subject to non-decreasing y.

    Implemented here rather than imported from sklearn.isotonic so this module needs
    nothing beyond numpy/pandas, and a plotting import can never abort an analysis whose
    CSVs are already written.
    """
    v, w, c = [], [], []
    for yi in np.asarray(y, float):
        v.append(yi); w.append(1.0); c.append(1)
        while len(v) > 1 and v[-2] > v[-1]:
            nv = (v[-1] * w[-1] + v[-2] * w[-2]) / (w[-1] + w[-2])
            nw, nc = w[-1] + w[-2], c[-1] + c[-2]
            del v[-2:], w[-2:], c[-2:]
            v.append(nv); w.append(nw); c.append(nc)
    return np.repeat(np.array(v), np.array(c))


def isotonic_auc_curve(u, pi):
    """Monotone fit of placement value on log10 burden.

    Monotone because the claim under test is directional (AUC non-decreasing in burden);
    a monotone fit cannot manufacture a non-monotonicity out of sampling noise.
    """
    m = np.isfinite(u) & np.isfinite(pi) & (pi > 0)
    if m.sum() < 5:
        return np.array([]), np.array([])
    x = np.log10(pi[m])
    o = np.argsort(x)
    return 10.0 ** x[o], _pava(u[m][o])


def detection_wall_pct(n_patches):
    """Burden (%) at which beta = 1/2, i.e. pi = N^(-1/2), for the median slide.

    Below this, the bag mean -- and any convex combination of instances, which includes
    attention pooling -- has asymptotically zero power, so it is the natural
    pre-registered split for AUC_low / AUC_high.
    """
    n = np.asarray(n_patches, float)
    n = n[np.isfinite(n) & (n > 1)]
    return float(np.median(n) ** -0.5 * 100.0) if len(n) else np.nan


# =====================================================================================
# Few-shot split without loading features
# =====================================================================================

def fewshot_split_ids(Y, k_shots, seed):
    """Reproduce utils.utils.fewshot_sampling's split, returning INDICES only.

    fewshot_sampling needs X (all the .npy bag features) just to slice it, and returns
    data rather than indices. MI-Zero never loads those features, so it reproduces the
    draw directly: fewshot_sampling re-seeds with np.random.seed(seed) itself, then
    draws k per class in class order, which makes it deterministic given (Y, k, seed)
    regardless of any earlier RNG use.

    analyze_tumor_ratio_miladapter.py cross-checks this against the real
    fewshot_sampling on live data, so a divergence surfaces immediately.
    """
    Y = np.asarray(Y)
    np.random.seed(seed)
    ids = np.arange(len(Y))
    train_ids = []
    for cls in range(len(np.unique(Y))):
        cls_ids = ids[Y == cls]
        train_ids.extend(np.random.choice(cls_ids, size=k_shots, replace=False))
    train_ids = np.array(train_ids)
    return train_ids, np.setdiff1d(ids, train_ids)


# =====================================================================================
# One evaluation of every metric
# =====================================================================================

def evaluate(pos_scores, pos_pi, neg_scores, edges, wall_pct):
    """All metrics for one seed / one bootstrap replicate. Flat dict."""
    u = placement_values(pos_scores, neg_scores)
    out = {"auc_overall": auc_from_placement(u)}

    idx = np.digitize(pos_pi, edges) - 1
    nz = pos_pi > 0
    for b in range(len(edges) - 1):
        m = (idx == b) & nz
        key = f"{edges[b]:g}-{edges[b+1]:g}"
        out[f"auc[{key}]"] = auc_from_placement(u[m]) if m.sum() else np.nan
        out[f"n[{key}]"] = int(m.sum())

    zero = ~nz
    out["auc[pi=0]"] = auc_from_placement(u[zero]) if zero.sum() else np.nan
    out["n[pi=0]"] = int(zero.sum())

    if np.isfinite(wall_pct):
        lo, hi = nz & (pos_pi < wall_pct), nz & (pos_pi >= wall_pct)
        out["auc_low"] = auc_from_placement(u[lo]) if lo.sum() else np.nan
        out["auc_high"] = auc_from_placement(u[hi]) if hi.sum() else np.nan
        out["auc_gap"] = out["auc_high"] - out["auc_low"]
        out["n_low"], out["n_high"] = int(lo.sum()), int(hi.sum())

    for spec in SPECS:
        s_all, thr = sens_at_spec(pos_scores, neg_scores, spec)
        out[f"sens@{spec:.2f}spec"] = s_all
        for b in range(len(edges) - 1):
            m = (idx == b) & nz
            key = f"{edges[b]:g}-{edges[b+1]:g}"
            out[f"sens@{spec:.2f}spec[{key}]"] = (
                float((pos_scores[m] > thr).mean()) if m.sum() else np.nan)

    out["slope_u_per_decade"], out["intercept_u"] = slope_u_vs_logpi(u, pos_pi)
    return out


def paired_bootstrap(runs, edges, wall_pct, n_boot=2000, seed=0):
    """Resample slides; the NEGATIVE POOL IS RESAMPLED ONCE PER REPLICATE and shared by
    every stratum and seed. The statistic is the seed-average, so the interval covers
    slide sampling only.

    runs: list of (name, pos_scores, pos_pi, neg_scores), all sharing a slide set.
    """
    rng = np.random.default_rng(seed)
    n_pos, n_neg = len(runs[0][1]), len(runs[0][3])
    reps = []
    for _ in range(n_boot):
        pi_idx = rng.integers(0, n_pos, n_pos)
        ng_idx = rng.integers(0, n_neg, n_neg)
        per_seed = [evaluate(ps[pi_idx], pp[pi_idx], ns[ng_idx], edges, wall_pct)
                    for _, ps, pp, ns in runs]
        rep = {}
        for k in per_seed[0]:
            if k.startswith("n[") or k.startswith("n_"):
                continue
            # a resample can empty a stratum, making every seed NaN for that key
            vals = [d[k] for d in per_seed if np.isfinite(d[k])]
            rep[k] = float(np.mean(vals)) if vals else np.nan
        reps.append(rep)
    return pd.DataFrame(reps)


# =====================================================================================
# Assembly
# =====================================================================================

def summarise(runs, edges, wall_pct, n_boot=2000, boot_seed=0):
    """Returns (bins_df, point, seed_sd, boot, summary)."""
    per_seed = [evaluate(ps, pp, ns, edges, wall_pct) for _, ps, pp, ns in runs]
    point = {k: float(np.nanmean([d[k] for d in per_seed])) for k in per_seed[0]}
    seed_sd = {k: float(np.nanstd([d[k] for d in per_seed])) for k in per_seed[0]}
    boot = paired_bootstrap(runs, edges, wall_pct, n_boot, boot_seed)

    pi_all = runs[0][2]
    idx = np.digitize(pi_all, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        key = f"{edges[b]:g}-{edges[b+1]:g}"
        m = (idx == b) & (pi_all > 0)
        row = {"bin_pct": key, "x": float(np.median(pi_all[m])) if m.sum() else np.nan,
               "n_tumor": int(m.sum()), "auc": point[f"auc[{key}]"],
               "auc_seed_sd": seed_sd[f"auc[{key}]"],
               "auc_lo": float(np.nanpercentile(boot[f"auc[{key}]"], 2.5)),
               "auc_hi": float(np.nanpercentile(boot[f"auc[{key}]"], 97.5))}
        for spec in SPECS:
            row[f"sens@{spec:.2f}spec"] = point[f"sens@{spec:.2f}spec[{key}]"]
        rows.append(row)
    bins_df = pd.DataFrame(rows)
    bins_df.loc[len(bins_df)] = {"bin_pct": "pi=0 (no patch registered)", "x": np.nan,
                                 "n_tumor": point["n[pi=0]"], "auc": point["auc[pi=0]"],
                                 "auc_seed_sd": seed_sd["auc[pi=0]"],
                                 "auc_lo": np.nan, "auc_hi": np.nan}

    summary = {"n_tumor": len(runs[0][1]), "n_normal": len(runs[0][3]),
               "n_runs": len(runs), "wall_pct": wall_pct, "bin_edges_pct": list(edges)}
    for k in (["auc_overall", "auc_low", "auc_high", "auc_gap", "slope_u_per_decade"]
              + [f"sens@{s:.2f}spec" for s in SPECS]):
        if k in point:
            summary[k] = {"value": point[k], "seed_sd": seed_sd.get(k),
                          "ci95": ([float(np.nanpercentile(boot[k], 2.5)),
                                    float(np.nanpercentile(boot[k], 97.5))]
                                   if k in boot else None)}
    return bins_df, point, seed_sd, boot, summary


def print_report(title, bins_df, summary):
    print("\n" + "=" * 88)
    print(f"Discrimination vs tumor ratio  [{title}]")
    print("=" * 88)
    print(f"{summary['n_tumor']} tumor / {summary['n_normal']} normal slides, "
          f"{summary['n_runs']} run(s); beta=1/2 wall at {summary['wall_pct']:.2f}% "
          f"of tissue patches")
    print(bins_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("-" * 88)
    for k in ["auc_overall", "auc_low", "auc_high", "auc_gap"]:
        if k in summary:
            s = summary[k]
            print(f"{k:<34} {s['value']:.4f}  [95% CI {s['ci95'][0]:.4f}, "
                  f"{s['ci95'][1]:.4f}]  seed SD {s['seed_sd']:.4f}")
    g = summary["slope_u_per_decade"]
    print(f"{'slope (AUC per decade of burden)':<34} {g['value']:+.4f}  "
          f"[95% CI {g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]  seed SD {g['seed_sd']:.4f}")
    print("=" * 88)
    print("CI is slide-sampling only; seed SD is the training component. They do not\n"
          "combine -- every seed sees the same slides. Quote auc_gap or the slope as the\n"
          "headline; the per-bin table is description, and at ~16 slides/bin a per-bin\n"
          "difference below about 0.15 is not resolvable.")


# =====================================================================================
# Figures
# =====================================================================================

def make_figure(bins_df, runs, edges, wall_pct, out_png, subtitle=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_AUC, C_S90, C_S95, C_WALL = "#1f5fa9", "#b5651d", "#8a8f98", "#555555"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.7))
    ok = bins_df.dropna(subset=["auc", "x"])

    axA.fill_between(ok["x"], ok["auc_lo"], ok["auc_hi"], color=C_AUC, alpha=.18, lw=0)
    axA.plot(ok["x"], ok["auc"], "o-", color=C_AUC, lw=2.2, ms=5)
    u_all = np.concatenate([placement_values(ps, ns) for _, ps, _, ns in runs])
    pi_all = np.concatenate([pp for _, _, pp, _ in runs])
    xs, ys = isotonic_auc_curve(u_all, pi_all)
    floor = 0.48
    if len(xs):
        axA.plot(xs, ys, "-", color="k", lw=1.3, alpha=.8)
        axA.annotate("isotonic fit of\nplacement values", (xs[-1], ys[-1]),
                     textcoords="offset points", xytext=(-4, -34), ha="right",
                     fontsize=7, color="0.25")
        floor = min(floor, float(np.nanmin(ys)))
    if len(ok):
        floor = min(floor, float(np.nanmin(ok["auc_lo"])))
    axA.axhline(0.5, color="0.6", lw=1, ls="--")
    if np.isfinite(wall_pct):
        axA.axvline(wall_pct, color=C_WALL, lw=1.1, ls=":")
        axA.text(wall_pct * 1.12, 0.53, r"$\beta=1/2$ wall", fontsize=7, color=C_WALL)
    axA.set_xscale("log"); axA.set_ylim(max(0.0, floor - 0.03), 1.02)
    axA.set_xlabel("tumor patch fraction (%)")
    axA.set_ylabel("AUC vs the full normal pool")
    axA.set_title("Discrimination against a common\nnegative pool, by tumor burden", loc="left")

    cols = [f"sens@{s:.2f}spec" for s in SPECS]
    if all(c in ok for c in cols):
        for i, (spec, c) in enumerate(zip(SPECS, (C_S90, C_S95))):
            axB.plot(ok["x"], ok[cols[i]], "o-", color=c, lw=1.9, ms=4.5,
                     label=f"{spec:.0%} specificity")
        # loc="best" rather than a fixed corner: these curves saturate at 1.0 whenever
        # the model separates well, so any hand-placed anchor collides in some runs.
        axB.legend(frameon=False, fontsize=7, loc="best")
    axB.set_xscale("log"); axB.set_ylim(0, 1.05)
    axB.set_xlabel("tumor patch fraction (%)")
    axB.set_ylabel("sensitivity at a fixed threshold")
    axB.set_title("Operating points, threshold set once\non the whole normal pool", loc="left")

    fig.text(0.005, 0.015, subtitle, fontsize=7, color="0.35")
    fig.subplots_adjust(wspace=0.32, bottom=0.26, top=0.84)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_sweep_figure(curves, wall_pct, out_png, legend_title, subtitle=""):
    """One AUC-vs-burden line per setting (e.g. per K), on shared axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    cmap = plt.get_cmap("viridis")
    n = max(len(curves), 2)
    for i, (label, bins_df) in enumerate(curves):
        ok = bins_df.dropna(subset=["auc", "x"])
        ax.plot(ok["x"], ok["auc"], "o-", color=cmap(0.08 + 0.84 * i / (n - 1)),
                lw=1.8, ms=4, label=str(label))
    ax.axhline(0.5, color="0.6", lw=1, ls="--")
    if np.isfinite(wall_pct):
        ax.axvline(wall_pct, color="0.4", lw=1.1, ls=":")
        # mid-height, not at the floor: the floor is where the legend wants to sit
        ax.text(wall_pct * 1.1, 0.62, r"$\beta=1/2$ wall", fontsize=7, color="0.4")
    ax.set_xscale("log"); ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("tumor patch fraction (%)")
    ax.set_ylabel("AUC vs the full normal pool")
    ax.set_title("Which setting wins where is\nthe whole question", loc="left")
    ax.legend(title=legend_title, frameon=False, fontsize=7, title_fontsize=7,
              loc="best", ncol=2)
    fig.text(0.005, 0.015, subtitle, fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================================================
# Self-test
# =====================================================================================

def self_test():
    print("self_test: tumor_ratio_metrics")
    rng = np.random.default_rng(0)

    pos, neg = rng.normal(1.0, 1, 500), rng.normal(0, 1, 800)
    u = placement_values(pos, neg)
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    ref = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    assert abs(auc_from_placement(u) - ref) < 1e-9
    assert abs(ref - 0.7602) < 0.03, ref
    assert abs(placement_values([0.0], [0.0, 0.0]).item() - 0.5) < 1e-12
    assert placement_values([1.0], [0.0, 0.0]).item() == 1.0

    # binary scores: AUC degenerates to balanced accuracy EXACTLY. This is the identity
    # that makes a saved pred_label useless for this analysis.
    bp, bn = (pos > 0.5).astype(float), (neg > 0.5).astype(float)
    bal = (bp.mean() + (1 - bn.mean())) / 2
    assert abs(auc_from_placement(placement_values(bp, bn)) - bal) < 1e-12
    print(f"  AUC(binary preds) == balanced accuracy ({bal:.6f}) exactly")

    edges = DEFAULT_EDGES
    pi = 10 ** rng.uniform(-1.7, 1.4, 300)
    s_pos = 0.8 * np.log10(pi) + 1.5 + rng.normal(0, 1, 300)
    s_neg = rng.normal(0, 1, 223)
    a = evaluate(s_pos, pi, s_neg, edges, 1.63)
    b = evaluate(s_pos * 3 - 4, pi, s_neg * 3 - 4, edges, 1.63)
    for k in [k for k in a if k.startswith("auc")]:
        if np.isfinite(a[k]):
            assert abs(a[k] - b[k]) < 1e-9, k
    sA = (s_pos > np.quantile(s_neg, 0.30)).mean()
    sB = (s_pos > np.quantile(s_neg, 0.97)).mean()
    assert sA - sB > 0.3
    print(f"  AUC invariant to monotone rescaling ({a['auc_overall']:.3f}); "
          f"sensitivity moves {sA:.2f} -> {sB:.2f}")
    assert a["slope_u_per_decade"] > 0.05 and a["auc_high"] > a["auc_low"]
    assert abs(detection_wall_pct([3775] * 11) - 1.6274) < 1e-3

    # few-shot split reproduction: k per class, disjoint, deterministic
    Y = np.array([0] * 239 + [1] * 160)
    tr, va = fewshot_split_ids(Y, 16, 3)
    assert len(tr) == 32 and len(np.intersect1d(tr, va)) == 0
    assert len(tr) + len(va) == len(Y)
    assert (Y[tr] == 0).sum() == 16 and (Y[tr] == 1).sum() == 16
    assert np.array_equal(tr, fewshot_split_ids(Y, 16, 3)[0])
    assert not np.array_equal(tr, fewshot_split_ids(Y, 16, 4)[0])
    print(f"  fewshot_split_ids: {len(tr)} support / {len(va)} eval, deterministic")

    runs = [("s0", s_pos, pi, s_neg), ("s1", s_pos + rng.normal(0, .2, 300), pi, s_neg)]
    bins_df, point, seed_sd, boot, summary = summarise(runs, edges, 1.63, n_boot=40)
    assert len(bins_df) == len(edges) and boot["auc_overall"].std() > 0
    print(f"  summarise: {len(bins_df)} rows, bootstrap SE {boot['auc_overall'].std():.4f}")
    print("self_test: OK")
