"""
utils/tail_proposition.py

Empirical check of the guiding proposition of ConceptTail MIL.

Setup: a normal slide has ~N_eff effectively independent patches (scores ~ F0); a tumor
slide has m tumor patches (scores ~ F1); slide score = top-K mean. The top of a normal
slide sits near the operating point beta* = K / N_eff, so

    P(detect) ~= P( Bin(m, TPR(beta*)) >= K )     (K = 1: 1 - (1 - TPR(1/N_eff))^m)

"detect" = the tumor slide's score beats a TYPICAL normal slide (the median normal slide
score), the event the approximation describes.

Refinement (pred, the default prediction). The formula above ignores two things that matter
at small m: (i) the tumor slide's own N - m normal patches also compete -- on their own they
beat the median normal slide half the time; (ii) the MEDIAN normal-slide top sits at FPR
beta_K = lambda_K / N_eff, where P(Poisson(lambda_K) >= K) = 1/2 (lambda_1 = ln 2), not K/N_eff.
Treating the top-K mean as the K-th largest score:
    P(detect) ~= P( Poisson(lambda_K) + Bin(m, TPR(beta_K)) >= K )
    K = 1:  1 - (1/2) (1 - TPR(ln2 / N_eff))^m
Both are reported: pred (refined) and pred_simple (the original statement). The excess over
chance, pred - 1/2, is still ~ linear in TPR at small m -- the proposition's message holds.

proposition_table() takes eval-set patch scores of ONE scorer and returns, per tumor slide:
    m                    annotated tumor patches
    pred                 refined prediction (above), TPR pooled over all eval tumor patches
    pred_simple          P(Bin(m, TPR(K / N_eff)) >= K), the original statement
    pred_slide           refined, with the slide's own TPR (heterogeneity across slides)
    observed             1 if S_tumor > median normal-slide S
    slide_auc            share of normal slides below S_tumor
and the scorer-level numbers: beta*, N_eff, TPR(beta*), normalised patch pAUC on [0, beta*]
and the full patch AUC (to show which one tracks slide sensitivity).

    python -m utils.tail_proposition     # self-test
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import binom, poisson


def tpr_at_fpr(normal_sorted, tumor_scores, beta):
    """Threshold = (1 - beta) quantile of the normal patch pool; share of tumor patches above."""
    n = len(normal_sorted)
    thr = normal_sorted[min(n - 1, int(np.ceil((1 - beta) * n)) - 1)] if beta > 0 else normal_sorted[-1]
    t = np.asarray(tumor_scores)
    return (float((t > thr).mean()) if len(t) else np.nan), float(thr)


def partial_auc(normal_sorted, tumor_scores, beta):
    """(1/beta) int_0^beta TPR(f) df  =  mean over tumor patches of (beta - FPR(t))_+ / beta."""
    t = np.asarray(tumor_scores)
    if not len(t):
        return np.nan
    fpr = 1.0 - np.searchsorted(normal_sorted, t, "right") / len(normal_sorted)
    return float(np.clip(beta - fpr, 0, None).mean() / beta)


def full_auc(normal_sorted, tumor_scores):
    t = np.asarray(tumor_scores)
    lt = np.searchsorted(normal_sorted, t, "left")
    le = np.searchsorted(normal_sorted, t, "right")
    return float(((lt + le) / 2 / len(normal_sorted)).mean()) if len(t) else np.nan


def p_detect(m, tpr, K):
    """Original statement: P(Bin(m, TPR) >= K)."""
    return float(binom.sf(K - 1, int(m), tpr)) if m > 0 and np.isfinite(tpr) else 0.0


def median_rate(K):
    """lambda_K with P(Poisson(lambda_K) >= K) = 1/2 (lambda_1 = ln 2)."""
    return float(brentq(lambda lam: poisson.sf(K - 1, lam) - 0.5, 1e-6, 10.0 * K + 10))


def p_detect_refined(m, tpr, K):
    """P(Poisson(lambda_K) + Bin(m, TPR) >= K): own normal patches + tumor patches."""
    lam = median_rate(K)
    if m <= 0 or not np.isfinite(tpr):
        return float(poisson.sf(K - 1, lam))
    j = np.arange(0, min(int(m), K) + 1)
    pj = binom.pmf(j, int(m), tpr)
    tail = np.where(j >= K, 1.0, poisson.sf(K - 1 - j, lam))
    return float((pj * tail).sum() + (binom.sf(K, int(m), tpr) if m > K else 0.0))


def proposition_table(normal_patch_scores, normal_S, tumor_patch_scores, tumor_masks, tumor_S, K=1,
                      neff_factor=1.0, beta=None, tumor_info=None, max_pool=2_000_000, seed=0):
    """normal_patch_scores: list of arrays (eval normal slides); normal_S: their slide scores;
    tumor_patch_scores / tumor_masks / tumor_S: eval tumor slides (masks = annotated tumor
    patches); tumor_info: optional list of dicts merged into the rows (e.g. tumor_area_pct).
    Returns (per-tumor-slide DataFrame, scorer-level dict)."""
    rng = np.random.default_rng(seed)
    pool = np.concatenate(normal_patch_scores)
    if len(pool) > max_pool:
        pool = rng.choice(pool, max_pool, replace=False)
    pool = np.sort(pool)
    n_eff = neff_factor * float(np.median([len(s) for s in normal_patch_scores]))
    beta = K / n_eff if beta is None else beta                  # original operating point
    beta_med = median_rate(K) / n_eff                             # median normal-slide top
    T = np.concatenate([s[g] for s, g in zip(tumor_patch_scores, tumor_masks) if g.any()])
    tpr_simple, _ = tpr_at_fpr(pool, T, beta)
    tpr, thr = tpr_at_fpr(pool, T, beta_med)
    Sn = np.sort(np.asarray(normal_S, float))
    med = float(np.median(Sn))
    rows = []
    for j, (s, g, S) in enumerate(zip(tumor_patch_scores, tumor_masks, tumor_S)):
        m = int(g.sum())
        tpr_j = float((s[g] > thr).mean()) if m else np.nan
        r = {"m": m, "pred": p_detect_refined(m, tpr, K), "pred_simple": p_detect(m, tpr_simple, K),
             "pred_slide": p_detect_refined(m, tpr_j, K),
             "tpr_slide": tpr_j, "observed": float(S > med),
             "slide_auc": float((np.searchsorted(Sn, S, "left") + np.searchsorted(Sn, S, "right")) / 2 / len(Sn))}
        if tumor_info is not None:
            r.update(tumor_info[j])
        rows.append(r)
    info = {"beta": beta, "beta_med": beta_med, "N_eff": n_eff, "K": K, "threshold": thr,
            "tpr_beta": tpr_simple, "tpr_beta_med": tpr,
            "pauc_beta": partial_auc(pool, T, beta), "patch_auc": full_auc(pool, T),
            "n_normal_patches": len(pool), "n_tumor_patches": len(T)}
    return pd.DataFrame(rows), info


def proposition_figure(by_bin, order, out_png, title, footer=""):
    """by_bin: rows method, bin, observed_mean, pred_mean. Left: observed (solid) and predicted
    (dashed) by bin, one colour per method; right: observed vs predicted per (method, bin)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#e87ba4", "#184f95", "#008300",
               "#a3261f", "#5598e7", "#9a6a00", "#9085e9", "#52514e", "#3d8f8f"]
    methods = list(dict.fromkeys(by_bin.method))
    col = {m: palette[i % len(palette)] for i, m in enumerate(methods)}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    x = np.arange(len(order))
    for mth in methods:
        g = by_bin[by_bin.method == mth].set_index("bin").reindex(order)
        ax.plot(x, g.observed_mean, "-o", color=col[mth], lw=2, ms=5)
        ax.plot(x, g.pred_mean, "--", color=col[mth], lw=1.3, alpha=0.8)
    ax.set_xticks(x, order)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("tumor area, % of tissue")
    ax.set_ylabel("tumor slides beating the median normal slide")
    ax.grid(axis="y", color="#e4e3df")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], color="#52514e", lw=2, marker="o", ms=5, label="observed"),
                       Line2D([], [], color="#52514e", lw=1.3, ls="--", label="predicted")],
              loc="lower right", frameon=False, fontsize=8)
    ax = axes[1]
    for mth in methods:
        g = by_bin[by_bin.method == mth]
        ax.scatter(g.pred_mean, g.observed_mean, color=col[mth], s=30)
    ax.plot([0, 1], [0, 1], color="#52514e", lw=1, ls=":")
    ax.set_xlabel("predicted  P(Poisson(λ_K) + Bin(m, TPR(β_K)) ≥ K)")
    ax.set_ylabel("observed")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_aspect("equal")
    ax.spines[["top", "right"]].set_visible(False)
    handles = [Line2D([], [], color=col[m], lw=2.5, label=m) for m in methods]
    ncol = min(3, len(methods))
    nrow = int(np.ceil(len(methods) / ncol))
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=11)
    if footer:
        import textwrap
        fig.text(0.01, -0.02, "\n".join(textwrap.wrap(footer, 190)), fontsize=7.5, color="#52514e", va="top")
    fig.tight_layout(rect=(0, 0.035 * nrow + 0.02, 1, 0.95))
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def self_test():
    print("self_test: utils.tail_proposition")
    rng = np.random.default_rng(0)
    N, shift = 2000, 3.0
    normals = [rng.standard_normal(N) for _ in range(300)]
    for K in (1, 3):
        nS = [np.sort(s)[-K:].mean() for s in normals]
        ts, gs, tS, info = [], [], [], []
        for m in [1, 2, 3, 5, 8, 15] * 60:
            s = rng.standard_normal(N)
            g = np.zeros(N, bool)
            g[:m] = True
            s[:m] = rng.standard_normal(m) + shift
            ts.append(s)
            gs.append(g)
            tS.append(np.sort(s)[-K:].mean())
            info.append({"m_group": m})
        df, inf = proposition_table(normals, nS, ts, gs, tS, K=K, tumor_info=info)
        by = df.groupby("m_group")[["pred_simple", "pred", "observed"]].mean()
        err = float((by.pred - by.observed).abs().max())
        print(f"  K={K}: beta* = {inf['beta']:.5f}, TPR(beta*) = {inf['tpr_beta']:.3f}, "
              f"max |pred - obs| over m = {err:.3f}")
        print("    " + by.T.to_string(float_format=lambda v: f"{v:.2f}").replace("\n", "\n    "))
        assert err < 0.15, by
    srt = np.sort(rng.standard_normal(100000))
    assert abs(partial_auc(srt, srt + 100, 0.01) - 1) < 1e-9 and partial_auc(srt, srt - 100, 0.01) == 0
    assert abs(full_auc(srt, rng.standard_normal(5000)) - 0.5) < 0.03
    print("self_test: OK")


if __name__ == "__main__":
    self_test()
