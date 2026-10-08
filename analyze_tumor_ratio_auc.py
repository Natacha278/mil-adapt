"""
analyze_tumor_ratio_auc.py

Discrimination-vs-tumor-ratio for CAMELYON16, replacing the sensitivity-only curve in
analyze_tumor_pct_accuracy.py.

WHY NOT analyze_tumor_pct_accuracy.py
-------------------------------------
That script runs with --tumor_only on, so every stratum has n_normal_gt = 0 and the
per-bin number is RECALL ON TUMOR SLIDES at whatever threshold the model happened to
pick. It cannot separate discrimination (does the model rank a low-burden tumor slide
above a normal slide?) from operating point (how readily does it say "Tumor"?). On this
evaluation split, one score vector read at two different thresholds -- identical
discrimination by construction -- gives sensitivity-vs-ratio curves differing by ~0.39
on average across bins, with different apparent slopes. That is enough to manufacture a
method difference that does not exist.

Excluding Normal slides from the BINNING is correct (they all sit at 0% by
construction). The fix is not to bin them, it is to use ALL of them as a common
reference pool:

    AUC_b = 1/(|T_b|*|N|) * sum_{i in T_b} sum_{j in N} [ 1(s_i > s_j) + 0.5*1(s_i = s_j) ]

the Mann-Whitney statistic of stratum b against the full normal pool N. Threshold-free,
and the denominator pool is the same in every stratum, so strata are mutually
comparable. This is a CONDITIONAL AUC (conditioning on the positive's tumor burden), not
a partial AUC and not the AUC of a restricted sub-problem.

WHAT THIS REPORTS
-----------------
1. Per-stratum AUC vs the full normal pool, with paired-bootstrap CIs.
2. Sensitivity at a GLOBALLY fixed specificity (90%, 95%): threshold chosen once on the
   whole normal pool, applied unchanged in every stratum. Keeps threshold-consistency
   while staying clinically readable.
3. Placement values and the binning-free headline. For tumor slide i,
       u_i = 1/|N| * sum_j [ 1(s_i > s_j) + 0.5*1(s_i = s_j) ]   in [0,1]
   is that slide's rank inside the normal pool, and E[u_i | stratum] is exactly that
   stratum's AUC. So u can be regressed directly on burden,
       u_i ~ alpha + gamma * log10(pi_i)
   giving ONE number with a CI -- "AUC falls by gamma per decade of tumor ratio" -- with
   no bin edges anywhere. An isotonic fit of u on log10(pi) gives the continuous AUC(pi)
   curve the bins approximate.
4. AUC_low / AUC_high either side of the beta = 1/2 detection wall, and their difference
   as a single pre-registered effect size.

VARIANCE: TWO COMPONENTS, DO NOT CONFLATE
-----------------------------------------
* Slide sampling -> paired bootstrap. The normal pool is resampled ONCE per replicate
  and shared by every stratum, because the strata share their negatives and their AUCs
  are therefore positively correlated. Resampling per stratum independently would give
  wrong intervals for DIFFERENCES between strata or methods.
* Training/seed randomness -> pass several --per_slide_csv files (one per seed). Seeds
  are summarised as a spread, NOT folded into the bootstrap.
Averaging over seeds does not shrink the slide-sampling variance: every seed is
evaluated on the same slides. The two are reported separately for that reason.

BIN EDGES ARE FIXED, NOT QUANTILES
----------------------------------
analyze_tumor_pct_accuracy.py defaults to pd.qcut, so edges move with whichever slides
landed in the run -- the committed files have (0.049, 0.098] / (0.048, 0.1] /
(0.1, 0.21] / (0.1, 0.23] / (0.1, 0.22] for nominally the same bin across runs. Methods
then get compared bin-by-bin on bins that are not the same bins, and a paired test
across methods is ill-defined. Default here is fixed, pre-registered edges.

INPUT
-----
One or more per-slide CSVs (one per seed) with at least:
    slide_id (or WSI), gt_label, score
`score` is the continuous slide-level tumor score -- a logit, a margin, or a
probability; only its ordering matters. main.py does not currently save one
(per_slide.csv stops at pred_label), so that is the blocking upstream change.
run_mizero.py writes CSVs in exactly this format already.
tumor_patch_pct is merged from local_data/csv/<project>.csv if absent.

USAGE
-----
    python analyze_tumor_ratio_auc.py --self_test
    python analyze_tumor_ratio_auc.py \
        --per_slide_csv analysis/mizero_k20_per_slide.csv \
        --out_dir analysis_auc --out_prefix CAMELYON16_MIZERO_k20
    # several seeds -> seed spread reported separately from the bootstrap CI
    python analyze_tumor_ratio_auc.py \
        --per_slide_csv analysis/run_seed*_per_slide.csv --out_prefix CAMELYON16_TaskRes_ABMIL

No GPU, no model loading, no feature access -- runs on a login node.
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# Fixed, pre-registered edges in PERCENT of tissue patches. Spaced roughly
# logarithmically because tumor burden is log-distributed, with 2.5% sitting near the
# beta = 1/2 wall for this patching configuration.
DEFAULT_EDGES = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 100.0]
SPECS = (0.90, 0.95)


# =====================================================================================
# Estimators
# =====================================================================================

def placement_values(pos, neg):
    """u_i = fraction of the negative pool that tumor slide i outranks (ties at 0.5).

    mean(u) over any subset is exactly that subset's AUC against `neg`, which is what
    lets the whole analysis run off a per-slide quantity. O((n+m) log(n+m)).
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(neg) == 0:
        return np.full(len(pos), np.nan)
    order = np.argsort(neg)
    s = neg[order]
    lt = np.searchsorted(s, pos, side="left")       # strictly less than pos
    le = np.searchsorted(s, pos, side="right")      # <= pos
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
    """OLS slope of placement value on log10(tumor fraction): AUC change per decade."""
    m = np.isfinite(u) & np.isfinite(pi) & (pi > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    x = np.log10(pi[m])
    b, a = np.polyfit(x, u[m], 1)
    return float(b), float(a)


def _pava(y):
    """Pool-adjacent-violators: least-squares fit subject to non-decreasing y.

    Implemented here rather than imported from sklearn.isotonic so this script has no
    dependency beyond numpy/pandas/matplotlib -- it is ~10 lines and it keeps a plotting
    import from being able to abort an analysis whose CSVs are already written.
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
    """Monotone (non-decreasing in burden) fit of placement value on log10 burden.

    Monotone rather than free-form because the claim under test is directional: AUC is
    expected to be non-decreasing in tumor burden. A monotone fit cannot manufacture a
    non-monotonicity out of sampling noise.
    """
    m = np.isfinite(u) & np.isfinite(pi) & (pi > 0)
    if m.sum() < 5:
        return np.array([]), np.array([])
    x = np.log10(pi[m])
    o = np.argsort(x)
    return 10.0 ** x[o], _pava(u[m][o])


def detection_wall_pct(n_patches):
    """Tumor fraction (%) at which beta = 1/2, i.e. pi = N^(-1/2), for the median slide.

    Below this the bag mean (and any convex combination of instances, which includes
    attention pooling) has asymptotically zero power, so it is the natural pre-registered
    split point for AUC_low / AUC_high.
    """
    n = np.asarray(n_patches, float)
    n = n[np.isfinite(n) & (n > 1)]
    return float(np.median(n) ** -0.5 * 100.0) if len(n) else np.nan


# =====================================================================================
# One evaluation of every metric, for a given set of slide scores
# =====================================================================================

def evaluate(pos_scores, pos_pi, neg_scores, edges, wall_pct):
    """All metrics for one seed / one bootstrap replicate. Returns a flat dict."""
    u = placement_values(pos_scores, neg_scores)
    out = {"auc_overall": auc_from_placement(u)}

    idx = np.digitize(pos_pi, edges) - 1
    for b in range(len(edges) - 1):
        m = (idx == b) & (pos_pi > 0)
        key = f"{edges[b]:g}-{edges[b+1]:g}"
        out[f"auc[{key}]"] = auc_from_placement(u[m]) if m.sum() else np.nan
        out[f"n[{key}]"] = int(m.sum())

    zero = pos_pi <= 0
    out["auc[pi=0]"] = auc_from_placement(u[zero]) if zero.sum() else np.nan
    out["n[pi=0]"] = int(zero.sum())

    nz = pos_pi > 0
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
            out[f"sens@{spec:.2f}spec[{key}]"] = float((pos_scores[m] > thr).mean()) if m.sum() else np.nan

    out["slope_u_per_decade"], out["intercept_u"] = slope_u_vs_logpi(u, pos_pi)
    return out


# =====================================================================================
# Loading
# =====================================================================================

def load_runs(paths, csv_path, project, split, score_col):
    """Returns (runs, wall_pct). runs = list of (name, pos_scores, pos_pi, neg_scores)."""
    ref = None
    if csv_path and os.path.exists(csv_path):
        ref = pd.read_csv(csv_path)
        ref = ref.rename(columns={c: "slide_id" for c in ("WSI", "wsi") if c in ref.columns})

    runs, wall = [], np.nan
    for path in paths:
        df = pd.read_csv(path)
        df = df.rename(columns={c: "slide_id" for c in ("WSI", "wsi") if c in df.columns})
        if score_col not in df.columns:
            raise SystemExit(
                f"{path}: no '{score_col}' column. This analysis needs the CONTINUOUS "
                f"slide-level score, not pred_label -- main.py must dump the logit or "
                f"softmax probability (run_mizero.py already writes one). Columns found: "
                f"{list(df.columns)}")
        if "tumor_patch_pct" not in df.columns:
            if ref is None or "tumor_patch_pct" not in ref.columns:
                raise SystemExit(
                    f"{path}: no tumor_patch_pct and none in {csv_path}. "
                    f"Run add_tumor_pct_to_csv.py first.")
            df = df.merge(ref[["slide_id", "tumor_patch_pct"]], on="slide_id", how="left")
        if split != "all" and "split" in df.columns:
            df = df[df["split"] == split]
        df = df.dropna(subset=[score_col, "gt_label"])

        if ref is not None and "n_patches" in ref.columns and not np.isfinite(wall):
            wall = detection_wall_pct(ref.loc[ref.slide_id.isin(df.slide_id), "n_patches"])
        if not np.isfinite(wall) and "n_patches" in df.columns:
            wall = detection_wall_pct(df["n_patches"])

        t = df[df.gt_label == "Tumor"]
        n = df[df.gt_label == "Normal"]
        runs.append((os.path.basename(path), t[score_col].to_numpy(),
                     t["tumor_patch_pct"].to_numpy(float), n[score_col].to_numpy()))
    return runs, wall


# =====================================================================================
# Bootstrap
# =====================================================================================

def paired_bootstrap(runs, edges, wall_pct, n_boot, seed=0):
    """Resample slides; the NEGATIVE POOL IS RESAMPLED ONCE PER REPLICATE and shared by
    every stratum and every seed, so strata stay correlated the way they really are.
    The statistic is the seed-average, so the CI covers slide sampling only.
    """
    rng = np.random.default_rng(seed)
    n_pos = len(runs[0][1])
    n_neg = len(runs[0][3])
    reps = []
    for _ in range(n_boot):
        pi_idx = rng.integers(0, n_pos, n_pos)
        ng_idx = rng.integers(0, n_neg, n_neg)
        per_seed = [evaluate(ps[pi_idx], pp[pi_idx], ns[ng_idx], edges, wall_pct)
                    for _, ps, pp, ns in runs]
        # A resample can leave a stratum empty, making every seed NaN for that key;
        # np.nanmean would warn and still return NaN, so handle it explicitly.
        rep = {}
        for k in per_seed[0]:
            if k.startswith("n[") or k.startswith("n_"):
                continue
            vals = [d[k] for d in per_seed if np.isfinite(d[k])]
            rep[k] = float(np.mean(vals)) if vals else np.nan
        reps.append(rep)
    return pd.DataFrame(reps)


# =====================================================================================
# Figure
# =====================================================================================

def make_figure(bins_df, point, boot, runs, edges, wall_pct, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_AUC, C_S90, C_S95, C_WALL = "#1f5fa9", "#b5651d", "#8a8f98", "#555555"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.6, 3.7))

    ok = bins_df.dropna(subset=["auc"])
    axA.fill_between(ok["x"], ok["auc_lo"], ok["auc_hi"], color=C_AUC, alpha=.18, lw=0)
    axA.plot(ok["x"], ok["auc"], "o-", color=C_AUC, lw=2.2, ms=5)
    u_all = np.concatenate([placement_values(ps, ns) for _, ps, _, ns in runs])
    pi_all = np.concatenate([pp for _, _, pp, _ in runs])
    xs, ys = isotonic_auc_curve(u_all, pi_all)
    if len(xs):
        axA.plot(xs, ys, "-", color="k", lw=1.3, alpha=.8)
        axA.annotate("isotonic fit of\nplacement values", (xs[-1], ys[-1]),
                     textcoords="offset points", xytext=(-4, -34), ha="right",
                     fontsize=7, color="0.25")
    axA.axhline(0.5, color="0.6", lw=1, ls="--")
    if np.isfinite(wall_pct):
        axA.axvline(wall_pct, color=C_WALL, lw=1.1, ls=":")
        axA.text(wall_pct * 1.12, 0.53, r"$\beta=1/2$ wall", fontsize=7, color=C_WALL)
    # Floor follows the data: the isotonic fit can dip below 0.5 at the sparse end and
    # clipping it would hide exactly the regime under study.
    floor = np.nanmin([0.48, float(np.nanmin(ys)) if len(ys) else 0.48,
                       float(np.nanmin(ok["auc_lo"])) if len(ok) else 0.48]) - 0.03
    axA.set_xscale("log"); axA.set_ylim(max(0.0, floor), 1.02)
    axA.set_xlabel("tumor patch fraction (%)")
    axA.set_ylabel("AUC vs the full normal pool")
    axA.set_title("Discrimination against a common\nnegative pool, by tumor burden", loc="left")

    # Labelled at the point of MAXIMUM separation between the two curves, not at the
    # right end where they converge and the labels would sit on top of each other.
    cols = [f"sens@{s:.2f}spec" for s in SPECS]
    if all(c in ok for c in cols):
        gap = (ok[cols[0]] - ok[cols[1]]).abs()
        j = int(np.nanargmax(gap.to_numpy())) if gap.notna().any() else len(ok) - 1
        # Keyed in the panel's empty upper-left rather than on the curves: these lines
        # converge at the right and run close together, so any on-curve label collides
        # with the other series' markers. Colour carries the cross-reference.
        for i, (spec, c) in enumerate(zip(SPECS, (C_S90, C_S95))):
            col = f"sens@{spec:.2f}spec"
            axB.plot(ok["x"], ok[col], "o-", color=c, lw=1.9, ms=4.5)
            axB.text(0.04, 0.96 - 0.09 * i, f"{spec:.0%} specificity", color=c, fontsize=7,
                     transform=axB.transAxes, va="top", ha="left", zorder=10)
    axB.set_xscale("log"); axB.set_ylim(0, 1.05)
    axB.set_xlabel("tumor patch fraction (%)")
    axB.set_ylabel("sensitivity at a fixed threshold")
    axB.set_title("Operating points, threshold set once\non the whole normal pool", loc="left")
    axB.margins(x=0.25)

    note = (f"{point['n_pos']} tumor / {point['n_neg']} normal slides; "
            f"{point['n_seeds']} seed(s); band = 95% paired bootstrap over slides")
    fig.text(0.005, 0.015, note, fontsize=7, color="0.35")
    fig.subplots_adjust(wspace=0.32, bottom=0.26, top=0.84)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =====================================================================================
# Self-test
# =====================================================================================

def self_test():
    print("self_test: estimators")
    rng = np.random.default_rng(0)

    # placement values reproduce a known AUC
    pos, neg = rng.normal(1.0, 1, 500), rng.normal(0, 1, 800)
    u = placement_values(pos, neg)
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    ref = (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    assert abs(auc_from_placement(u) - ref) < 1e-9, (auc_from_placement(u), ref)
    assert abs(ref - 0.7602) < 0.03, ref                       # Phi(1/sqrt(2))

    # ties get exactly 0.5 credit
    assert abs(placement_values([0.0], [0.0, 0.0]).item() - 0.5) < 1e-12
    assert placement_values([1.0], [0.0, 0.0]).item() == 1.0
    assert placement_values([-1.0], [0.0, 0.0]).item() == 0.0

    # THE POINT OF THE SCRIPT: AUC is invariant to the threshold, sensitivity is not
    edges = DEFAULT_EDGES
    pi = 10 ** rng.uniform(-1.7, 1.4, 300)
    s_pos = 0.8 * np.log10(pi) + 1.5 + rng.normal(0, 1, 300)
    s_neg = rng.normal(0, 1, 223)
    mono_shift = lambda s: s * 3.0 - 4.0                       # strictly increasing
    a = evaluate(s_pos, pi, s_neg, edges, 1.63)
    b = evaluate(mono_shift(s_pos), pi, mono_shift(s_neg), edges, 1.63)
    for k in [k for k in a if k.startswith("auc")]:
        if np.isfinite(a[k]):
            assert abs(a[k] - b[k]) < 1e-9, (k, a[k], b[k])
    print(f"  AUC invariant under monotone rescaling (overall {a['auc_overall']:.3f})")

    # a sensitivity curve read at two thresholds of the SAME scores diverges
    sA = (s_pos > np.quantile(s_neg, 0.30)).mean()
    sB = (s_pos > np.quantile(s_neg, 0.97)).mean()
    assert sA - sB > 0.3, (sA, sB)
    print(f"  same scores, two thresholds: sensitivity {sA:.2f} vs {sB:.2f}; "
          f"AUC identical at {a['auc_overall']:.3f}")

    # burden gradient is recovered
    assert a["slope_u_per_decade"] > 0.05, a["slope_u_per_decade"]
    assert a["auc_high"] > a["auc_low"], (a["auc_low"], a["auc_high"])
    print(f"  slope {a['slope_u_per_decade']:.3f}/decade, "
          f"AUC_low {a['auc_low']:.3f} -> AUC_high {a['auc_high']:.3f}")

    # wall and bootstrap plumbing
    assert abs(detection_wall_pct([3775] * 11) - 1.6274) < 1e-3, detection_wall_pct([3775]*11)
    runs = [("s0", s_pos, pi, s_neg), ("s1", s_pos + rng.normal(0, .2, 300), pi, s_neg)]
    bt = paired_bootstrap(runs, edges, 1.63, n_boot=40, seed=1)
    assert bt["auc_overall"].std() > 0 and len(bt) == 40
    print(f"  bootstrap SE(auc_overall) = {bt['auc_overall'].std():.4f}")
    print("self_test: OK")


# =====================================================================================
# Main
# =====================================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per_slide_csv", nargs="+",
                   help="One per-slide CSV per seed (globs allowed). Needs slide_id/WSI, "
                        "gt_label and a continuous score column.")
    p.add_argument("--score_col", default="score")
    p.add_argument("--csv_path", default=None,
                   help="Project CSV for tumor_patch_pct / n_patches. "
                        "Defaults to ./local_data/csv/<project>.csv.")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--split", choices=["val", "train", "all"], default="val")
    p.add_argument("--bin_edges", type=float, nargs="+", default=DEFAULT_EDGES,
                   help="FIXED bin edges in percent. Fixed rather than quantile so the "
                        "x-axis is identical across methods and seeds.")
    p.add_argument("--wall_pct", type=float, default=None,
                   help="Tumor %% splitting AUC_low/AUC_high. Default: computed as "
                        "median(n_patches)^-0.5, the beta=1/2 point for this patching.")
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--boot_seed", type=int, default=0)
    p.add_argument("--out_dir", default="./analysis_auc")
    p.add_argument("--out_prefix", default=None)
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.per_slide_csv:
        p.error("--per_slide_csv is required (or use --self_test)")

    paths = sorted({q for pat in args.per_slide_csv for q in (glob.glob(pat) or [pat])})
    csv_path = args.csv_path or f"./local_data/csv/{args.project}.csv"
    runs, wall_auto = load_runs(paths, csv_path, args.project, args.split, args.score_col)
    wall = args.wall_pct if args.wall_pct is not None else wall_auto
    edges = list(args.bin_edges)
    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.out_prefix or f"{args.project}_{args.split}"

    per_seed = [evaluate(ps, pp, ns, edges, wall) for _, ps, pp, ns in runs]
    point = {k: float(np.nanmean([d[k] for d in per_seed])) for k in per_seed[0]}
    seed_sd = {k: float(np.nanstd([d[k] for d in per_seed])) for k in per_seed[0]}
    boot = paired_bootstrap(runs, edges, wall, args.n_boot, args.boot_seed)
    point.update(n_pos=len(runs[0][1]), n_neg=len(runs[0][3]), n_seeds=len(runs),
                 wall_pct=wall)

    rows = []
    for b in range(len(edges) - 1):
        key = f"{edges[b]:g}-{edges[b+1]:g}"
        pi_all = runs[0][2]
        m = (np.digitize(pi_all, edges) - 1 == b) & (pi_all > 0)
        row = {"bin_pct": key, "x": float(np.median(pi_all[m])) if m.sum() else np.nan,
               "n_tumor": int(m.sum()),
               "auc": point[f"auc[{key}]"], "auc_seed_sd": seed_sd[f"auc[{key}]"],
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

    summary = {
        "n_tumor": point["n_pos"], "n_normal": point["n_neg"], "n_seeds": point["n_seeds"],
        "wall_pct": wall, "split": args.split, "score_col": args.score_col,
        "inputs": paths, "bin_edges_pct": edges,
    }
    for k in ["auc_overall", "auc_low", "auc_high", "auc_gap", "slope_u_per_decade"] + \
             [f"sens@{s:.2f}spec" for s in SPECS]:
        if k in point:
            summary[k] = {"value": point[k], "seed_sd": seed_sd.get(k),
                          "ci95": [float(np.nanpercentile(boot[k], 2.5)),
                                   float(np.nanpercentile(boot[k], 97.5))] if k in boot else None}

    bins_csv = os.path.join(args.out_dir, f"{prefix}_bin_auc.csv")
    json_path = os.path.join(args.out_dir, f"{prefix}_summary.json")
    png = os.path.join(args.out_dir, f"{prefix}_auc_vs_tumor_pct.png")
    bins_df.to_csv(bins_csv, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    make_figure(bins_df.iloc[:-1], point, boot, runs, edges, wall, png)

    g = summary["slope_u_per_decade"]
    print("\n" + "=" * 86)
    print(f"Discrimination vs tumor ratio  [{prefix}]")
    print("=" * 86)
    print(f"{point['n_pos']} tumor / {point['n_neg']} normal slides, split={args.split}, "
          f"{point['n_seeds']} seed(s); beta=1/2 wall at {wall:.2f}% of tissue patches")
    print(bins_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("-" * 86)
    for k in ["auc_overall", "auc_low", "auc_high", "auc_gap"]:
        if k in summary:
            s = summary[k]
            print(f"{k:<16} {s['value']:.4f}  [95% CI {s['ci95'][0]:.4f}, {s['ci95'][1]:.4f}]"
                  f"  seed SD {s['seed_sd']:.4f}")
    print(f"{'slope (AUC/decade of tumor ratio)':<16} {g['value']:+.4f} "
          f" [95% CI {g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}]  seed SD {g['seed_sd']:.4f}")
    print("=" * 86)
    print(f"bins -> {bins_csv}\nsummary -> {json_path}\nfigure -> {png}")
    print("\nThe CI is slide-sampling only; seed SD is the training component. They do not\n"
          "combine -- every seed is evaluated on the same slides, so averaging seeds does\n"
          "not shrink the CI. Quote auc_gap (or the slope) as the headline effect size;\n"
          "the per-bin table is description, and at ~16 slides/bin a per-bin difference\n"
          "below about 0.15 is not resolvable.")


if __name__ == "__main__":
    main()
