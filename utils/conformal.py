"""
utils/conformal.py

Specificity-controlled slide threshold (step 4 of Text-TopK MIL) by leave-one-normal-out
conformal calibration. Model-agnostic: works with any fit / score functions (the numpy
and torch Text-TopK, or anything that returns one score per slide, higher = more tumor).

Why not threshold on the support normals directly: the model was trained to push their
scores down, so their in-sample scores are optimistic and a threshold set from them lets
more new normal slides through than intended. Eval normals cannot be used (label leak).

Procedure (few-shot support = n normals + m tumors)
  1. For each support normal j: fit a model m_-j on the support WITHOUT normal j (every
     tumor slide is kept in every fit), and score the held-out normal: z_j = S_-j(normal j).
  2. For a new slide x, compare it with each held-out normal under the model that left
     that normal out:
         count(x) = #{ j : S_-j(x) <= z_j }          (ties count against x)
         p(x)     = (1 + count(x)) / (n + 1)
  3. Predict Tumor iff p(x) <= alpha, i.e. iff count(x) <= c_max = floor(alpha (n+1) - 1).

If the normal slides are exchangeable, P(p(x) <= alpha | x normal) is about alpha (the
jackknife+ / cross-conformal argument; for one shared model it is exactly
(c_max + 1)/(n + 1) <= alpha). So specificity on new normal slides is ~ 1 - alpha by
construction, whatever the model.

What n normals allow: the smallest p-value is 1/(n+1), so alpha >= 1/(n+1). With 16
normals: alpha in [1/17, 2/17) -> Tumor iff x beats ALL 16 held-out normals;
alpha in [2/17, 3/17) -> all but at most one; etc. Specificity targets above 1 - 1/17
(~0.94) need more normals.

margin(x) = mean_j (S_-j(x) - z_j) is a continuous score in the same units as S (for
AUCs and plots; the decision uses count/p only).

    python -m utils.conformal          # self-test (numpy only, ~5 s)
"""

import math

import numpy as np


def count_max(n, alpha):
    """Largest count(x) still predicted Tumor: floor(alpha (n+1) - 1). Raises if alpha is
    below the smallest attainable p-value 1/(n+1)."""
    c = math.floor(alpha * (n + 1) - 1 + 1e-9)
    if c < 0:
        raise ValueError(f"alpha={alpha:g} is below 1/(n+1)={1 / (n + 1):.4f} with n={n} normal "
                         f"slides: nothing could ever be predicted Tumor. Use alpha >= "
                         f"{1 / (n + 1):.4f} or more normals.")
    return c


def nominal_level(n, alpha):
    """The false-positive rate the rule actually targets: (c_max + 1)/(n + 1) <= alpha."""
    return (count_max(n, alpha) + 1) / (n + 1)


def loo_normal_fits(bags, labels, fit_fn, score_fn, verbose=False):
    """fit_fn(bags, labels) -> fitted model; score_fn(model, H) -> float.
    Returns (models, z, normal_index): one model per support normal, fitted without it,
    and that normal's held-out score under it."""
    labels = np.asarray(labels)
    normals = np.where(labels == 0)[0]
    if len(normals) < 1 or (labels == 1).sum() < 1:
        raise ValueError("need at least one normal and one tumor support slide")
    models, z = [], []
    for c, j in enumerate(normals):
        keep = np.where(np.arange(len(labels)) != j)[0]
        m = fit_fn([bags[i] for i in keep], labels[keep])
        models.append(m)
        z.append(float(score_fn(m, bags[j])))
        if verbose:
            print(f"  LOO fit {c + 1}/{len(normals)}: held-out normal score {z[-1]:.3f}")
    return models, np.asarray(z, float), normals


class ConformalDetector:
    """Decision rule from loo_normal_fits. score_fn(model, H) -> float."""

    def __init__(self, models, z, score_fn, alpha=0.1):
        self.models, self.z, self.score_fn = list(models), np.asarray(z, float), score_fn
        self.n = len(self.z)
        self.set_alpha(alpha)

    def set_alpha(self, alpha):
        self.alpha, self.c_max = alpha, count_max(self.n, alpha)
        return self

    @classmethod
    def fit(cls, bags, labels, fit_fn, score_fn, alpha=0.1, verbose=False):
        count_max(int((np.asarray(labels) == 0).sum()), alpha)        # fail before training
        models, z, _ = loo_normal_fits(bags, labels, fit_fn, score_fn, verbose)
        return cls(models, z, score_fn, alpha)

    # ------------------------------------------------------------------ per slide
    def scores(self, H):
        """S_-j(x) for every LOO model j, shape (n,)."""
        return np.array([float(self.score_fn(m, H)) for m in self.models])

    def evaluate(self, H):
        """All quantities from one pass over the n models."""
        s = self.scores(H)
        c = int(np.sum(s <= self.z))
        return {"count": c, "p_value": (1 + c) / (self.n + 1),
                "margin": float(np.mean(s - self.z)), "pred": int(c <= self.c_max)}

    def p_value(self, H):
        return self.evaluate(H)["p_value"]

    def margin(self, H):
        return self.evaluate(H)["margin"]

    def predict(self, H):
        return self.evaluate(H)["pred"]

    def summary(self):
        return {"n_normals": self.n, "alpha": self.alpha, "c_max": self.c_max,
                "nominal_fpr": (self.c_max + 1) / (self.n + 1),
                "z_mean": float(self.z.mean()), "z_max": float(self.z.max())}


# =====================================================================================
# Self-test
# =====================================================================================

def self_test():
    print("self_test: utils.conformal")

    # --- 1. count / alpha bookkeeping ---------------------------------------------------
    assert count_max(16, 0.1) == 0 and count_max(16, 1 / 17) == 0
    assert count_max(16, 0.15) == 1 and count_max(16, 2 / 17) == 1 and count_max(16, 0.2) == 2
    for bad in (0.05, 1 / 17 - 1e-6):
        try:
            count_max(16, bad)
            raise AssertionError("alpha below 1/(n+1) accepted")
        except ValueError:
            pass
    assert abs(nominal_level(16, 0.1) - 1 / 17) < 1e-12
    print("  c_max: n=16 -> alpha 0.1: beat all 16, alpha 0.15: all but 1; alpha < 1/17 refused")

    # --- 2. hand-checked p-values with stub models ------------------------------------
    # model j scores a slide as H[j]; held-out normal scores z = [1, 2, 3]
    det = ConformalDetector([0, 1, 2], [1.0, 2.0, 3.0], lambda m, H: H[m], alpha=0.5)
    e = det.evaluate(np.array([0.5, 2.0, 3.5]))      # loses to z0, ties z1 (counts), beats z2
    assert e["count"] == 2 and e["p_value"] == 3 / 4 and e["pred"] == 0
    assert abs(e["margin"] - np.mean([-0.5, 0.0, 0.5])) < 1e-12
    e = det.evaluate(np.array([5.0, 5.0, 2.9]))      # beats two, loses to z2
    assert e["count"] == 1 and e["p_value"] == 2 / 4 and e["pred"] == 1   # c_max = 1 at alpha .5
    print("  p-value / count / margin match hand computation (ties count against the slide)")

    # --- 3. LOO fits really leave the normal out ---------------------------------------
    seen = []
    bags = [np.full((3, 2), i, float) for i in range(6)]
    labels = np.array([0, 0, 1, 0, 1, 1])
    models, z, normals = loo_normal_fits(
        bags, labels, lambda b, y: (seen.append(sorted(int(B[0, 0]) for B in b)), y.sum())[1],
        lambda m, H: float(H[0, 0]))
    assert list(normals) == [0, 1, 3] and list(z) == [0.0, 1.0, 3.0]
    assert seen == [[1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 2, 4, 5]]
    assert all(m == 3 for m in models)                           # every tumor kept
    print("  loo_normal_fits: each fit omits exactly its normal, keeps every tumor")

    # --- 4. specificity: conformal holds the target, the in-sample threshold does not --
    # A learner that adapts to its training normals (as Text-TopK does): it moves the
    # direction away from the training normals' top patches, so those score lower
    # in-sample than a new normal would.
    rng = np.random.default_rng(0)
    d, n_p, n_sup = 24, 120, 16
    w0 = np.eye(d)[0]

    def slide(tumor):
        H = rng.standard_normal((n_p, d))
        if tumor:
            H[:3] += 2.5 * w0
        return H

    def fit(b, y):
        top = np.array([H[np.argmax(H @ w0)] for H in b])
        w = w0 + 0.6 * (top[y == 1].mean(0) - top[y == 0].mean(0))
        return w / np.linalg.norm(w)

    def score(w, H):
        return float((H @ w).max())

    alpha, reps = 0.15, 150
    fpr_conf, fpr_ins = [], []
    for _ in range(reps):
        b = [slide(False) for _ in range(n_sup)] + [slide(True) for _ in range(n_sup)]
        y = np.array([0] * n_sup + [1] * n_sup)
        det = ConformalDetector.fit(b, y, fit, score, alpha=alpha)
        full = fit(b, y)
        ins = np.sort([score(full, H) for H in b[:n_sup]])
        thr = ins[-(det.c_max + 1)]                 # same order statistic, in-sample
        new = [slide(False) for _ in range(60)]
        fpr_conf.append(np.mean([det.predict(H) for H in new]))
        fpr_ins.append(np.mean([score(full, H) > thr for H in new]))
    fc, fi = float(np.mean(fpr_conf)), float(np.mean(fpr_ins))
    nominal = nominal_level(n_sup, alpha)
    se = float(np.std(fpr_conf) / np.sqrt(reps))
    print(f"  new normal slides flagged Tumor ({reps} cohorts, alpha {alpha}, nominal "
          f"{nominal:.3f}): conformal {fc:.3f} ± {se:.3f}, in-sample threshold {fi:.3f}")
    assert fc <= alpha + 3 * se, (fc, alpha)
    assert fi > fc + 0.02, (fi, fc)
    print("self_test: OK")


if __name__ == "__main__":
    self_test()
