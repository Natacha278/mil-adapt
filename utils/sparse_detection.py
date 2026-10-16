"""
utils/sparse_detection.py

Slide-level pooling as sparse-mixture detection: patch scores -> patch p-values against
the NORMAL patch distribution -> one slide statistic that adapts to the (unknown) number
of tumor patches.

1. Patch p-values (empirical null)
   null = every patch score of the normal support slides (~10^5 patches at 16 shots).
       p_i = (1 + #{null >= s_i}) / (n0 + 1)
   p_i is how rare patch i's score is among normal patches (conformal patch p-value:
   exact if the patch were an exchangeable normal patch; the smallest value is 1/(n0+1)).

2. Slide statistics on the sorted p-values p_(1) <= ... <= p_(N) of one slide, searched
   over the most suspicious fraction i <= alpha0 * N only:
     HC   Higher Criticism (Donoho & Jin 2004)
              max_i  sqrt(N) (i/N - p_(i)) / sqrt(p_(i) (1 - p_(i)))
          "how many more small p-values than a normal slide would have", standardised.
     HC+  the same restricted to p_(i) > 1/N (the usual stabilised variant; ignores the
          most extreme p-values, so weaker for 1-3 tumor patches).
     BJ   Berk-Jones:  max_{i: p_(i) < i/N}  N * KL( i/N || p_(i) )
          (binomial log-likelihood ratio; more powerful than HC in the very sparse regime)
     min-p    -log10 p_(1)  (the max patch score; Tippett)
   and, on raw scores, the fixed rules they replace: top-K mean for each K, mean.

   Why adaptive: with k tumor patches out of N, top-1 is optimal when k is 1-2, a top-K
   mean when k ~ K, the mean when k is a large fraction. HC/BJ take the max over all i,
   so they pick the best "k" per slide -- no K to tune per lesion size.

   Not assumed: independence. Patches of one slide are correlated and slides shift
   globally (stain, scanner), so HC/BJ of normal slides do not follow their theoretical
   null -- the slide threshold must still be calibrated on normal slides (conformal
   step or a specificity quantile), and AUCs are unaffected.

    python -m utils.sparse_detection     # self-test: hand checks + sparse-mixture benchmark
"""

import numpy as np

STATS = ("min-p", "HC", "HC+", "BJ")


def patch_pvalues(null_sorted, scores):
    """null_sorted: ascending normal patch scores (n0); returns p for each score."""
    n0 = len(null_sorted)
    ge = n0 - np.searchsorted(null_sorted, np.asarray(scores, float), "left")
    return (1.0 + ge) / (n0 + 1.0)


def _kl_bern(a, b):
    """KL(Bern(a) || Bern(b)), elementwise, 0 log 0 = 0."""
    a = np.clip(a, 0, 1); b = np.clip(b, 1e-300, 1 - 1e-16)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(a > 0, a * np.log(a / b), 0.0)
        t2 = np.where(a < 1, (1 - a) * np.log((1 - a) / (1 - b)), 0.0)
    return t1 + t2


def sparse_statistics(p, alpha0=0.1):
    """p: patch p-values of one slide -> {min-p, HC, HC+, BJ}."""
    p = np.sort(np.asarray(p, float))
    N = len(p)
    m = max(1, int(np.ceil(alpha0 * N)))
    pi = np.clip(p[:m], 1e-300, 1 - 1e-12)
    frac = np.arange(1, m + 1) / N
    hc = np.sqrt(N) * (frac - pi) / np.sqrt(pi * (1 - pi))
    plus = pi > 1.0 / N
    above = frac > pi
    bj = N * _kl_bern(frac, pi)
    return {"min-p": float(-np.log10(p[0])),
            "HC": float(hc.max()),
            "HC+": float(hc[plus].max()) if plus.any() else 0.0,
            "BJ": float(bj[above].max()) if above.any() else 0.0}


def score_statistics(s, ks=(1, 10)):
    """Fixed pooling rules on raw patch scores: top-K means and the mean."""
    s = np.asarray(s, float)
    out = {}
    for k in ks:
        kk = max(1, min(int(np.ceil(k * len(s))) if 0 < k < 1 else int(k), len(s)))
        out[f"top-{k:g}"] = float(np.partition(s, len(s) - kk)[len(s) - kk:].mean())
    out["mean"] = float(s.mean())
    return out


def all_statistics(s, null_sorted, ks=(1, 10), alpha0=0.1):
    return {**score_statistics(s, ks), **sparse_statistics(patch_pvalues(null_sorted, s), alpha0)}


# =====================================================================================
# Self-test
# =====================================================================================

def _auc(pos, neg):
    neg = np.sort(neg)
    return float(np.mean((np.searchsorted(neg, pos, "left") + np.searchsorted(neg, pos, "right"))
                         / 2 / len(neg)))


def self_test():
    print("self_test: utils.sparse_detection")
    # --- 1. p-values by hand -------------------------------------------------------------
    null = np.sort(np.array([0.0, 1.0, 2.0, 3.0]))
    assert np.allclose(patch_pvalues(null, [3.5, 3.0, 1.5, -1.0]), [1 / 5, 2 / 5, 3 / 5, 5 / 5])
    print("  patch p-values: (1 + #null >= s) / (n0 + 1), ties count against the patch")

    # --- 2. HC / BJ by brute force ---------------------------------------------------------
    p = np.array([0.001, 0.02, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.4, 0.6])
    st = sparse_statistics(p, alpha0=0.3)                     # m = 3: i = 1, 2, 3
    ps, N = np.sort(p), 10
    hc = max(np.sqrt(N) * (i / N - ps[i - 1]) / np.sqrt(ps[i - 1] * (1 - ps[i - 1])) for i in (1, 2, 3))
    bj = max(N * (i / N * np.log(i / N / ps[i - 1]) + (1 - i / N) * np.log((1 - i / N) / (1 - ps[i - 1])))
             for i in (1, 2, 3) if i / N > ps[i - 1])
    hcp = max(np.sqrt(N) * (i / N - ps[i - 1]) / np.sqrt(ps[i - 1] * (1 - ps[i - 1]))
              for i in (1, 2, 3) if ps[i - 1] > 1 / N)
    assert abs(st["HC"] - hc) < 1e-9 and abs(st["BJ"] - bj) < 1e-9 and abs(st["HC+"] - hcp) < 1e-9
    assert abs(st["min-p"] - 3.0) < 1e-12
    sc = score_statistics(np.array([5.0, 1.0, 3.0, 2.0]), ks=(1, 2, 0.5))
    assert sc["top-1"] == 5.0 and sc["top-2"] == 4.0 and sc["top-0.5"] == 4.0 and sc["mean"] == 2.75
    print("  HC, HC+, BJ, min-p, top-K, mean match brute-force formulas")

    # --- 3. adaptivity: classic sparse-mixture benchmark (exact Gaussian null) -----------
    # Slides of N iid N(0,1) patch scores; tumor slides have k patches shifted by mu.
    # top-1 should win when k is tiny, the mean when k is large; HC/BJ should never be far
    # from the best rule in any regime.
    rng = np.random.default_rng(0)
    N, n_sl = 4000, 120
    null = np.sort(rng.standard_normal(200_000))
    regimes = [("very sparse  k=2,   mu=5.0", 2, 5.0), ("sparse       k=20,  mu=2.6", 20, 2.6),
               ("moderate     k=200, mu=0.9", 200, 0.9), ("dense        k=1600,mu=0.22", 1600, 0.22)]
    rules = ["top-1", "top-10", "mean", "HC", "HC+", "BJ"]
    print(f"  {'regime':28s}" + "".join(f"{r:>8s}" for r in rules))
    worst_gap = {}
    for name, k, mu in regimes:
        stats = {r: ([], []) for r in rules}
        for lab in (0, 1):
            for _ in range(n_sl):
                s = rng.standard_normal(N)
                if lab:
                    s[:k] += mu
                a = all_statistics(s, null, ks=(1, 10), alpha0=0.5)
                for r in rules:
                    stats[r][lab].append(a[r])
        aucs = {r: _auc(np.array(stats[r][1]), np.array(stats[r][0])) for r in rules}
        best = max(aucs.values())
        for r in rules:
            worst_gap[r] = max(worst_gap.get(r, 0.0), best - aucs[r])
        print(f"  {name:28s}" + "".join(f"{aucs[r]:8.3f}" for r in rules))
    print("  worst gap to the best rule over regimes: "
          + ", ".join(f"{r} {worst_gap[r]:.3f}" for r in rules))
    assert min(worst_gap["HC"], worst_gap["BJ"]) < min(worst_gap["top-1"], worst_gap["top-10"],
                                                       worst_gap["mean"]), worst_gap
    print("self_test: OK")


if __name__ == "__main__":
    self_test()
