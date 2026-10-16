"""
utils/witness_pseudolabel.py

Witness-rate-aware pseudo-labelling for few-shot Text-TopK MIL.

The ceiling analysis shows the bottleneck is supervision: a linear scorer trained on
PATCH labels of the same support slides is much better than one trained from slide
labels with a fixed top-K. This module turns slide labels into patch labels:

  1. Estimate each support tumor slide's number of tumor patches from the excess of
     small patch p-values against normal tissue (sparse-mixture / Meinshausen-Rice style
     lower bound):
         p_i      = (1 + #{null >= s_i}) / (n0 + 1)        null = normal support patches
         n_hat_j  = max(1, max_t [ #{p_i <= t} - N_j t - z sqrt(N_j t (1 - t)) ])   (z=3, t_max=0.05: z=2 over a 48-point grid gave false excess ~15 on null slides)
     t on a log grid from 1/(n0+1) to t_max. A normal slide has ~N t patches with p <= t;
     the excess beyond z binomial SDs is a conservative count of non-normal patches.
     The null must not have been trained on: round 0 uses the text prior alone (zero-shot,
     untrained); later rounds use 2-fold cross-fitted models (each trained without one
     half of the support normals, whose patches are its null).
  2. Pseudo-positives: the top n_hat_j patches of each support tumor slide under the
     current scorer (capped at cap_frac * N_j) -- 1-2 patches for an isolated-cell slide,
     hundreds for a macro-metastasis, instead of a fixed K.
  3. Patch-level training: BCE pushes pseudo-positives up (mean per slide, then over
     slides, so large lesions do not dominate) and EVERY normal-slide patch down (certain
     negatives), on top of the existing objective (slide BCE on top-K + text prior
     + optional hard-normal ranking):
         L = BCE_slide + lam ||Delta||^2 + mu L_rank + eta * 0.5 (L_pos + L_neg)
  4. EM-style rounds: refit from scratch with the new pseudo-labels, re-estimate, reselect.

Prediction is unchanged (top-K slide score, S > 0), so results are directly comparable.

Diagnostics (when patch ground truth is given for the support slides): per round and
slide, n_hat vs true number of tumor patches, and precision / recall of the selected
pseudo-positives, with the same numbers for a fixed top-K selection as reference.

    python -m utils.witness_pseudolabel      # self-test (CPU, ~1 min)
"""

import numpy as np
import torch
import torch.nn.functional as F

from utils.sparse_detection import patch_pvalues
from utils.text_topk_mil_torch import normal_tail_rank_loss, slide_bce_loss


# =====================================================================================
# 1. Witness estimate
# =====================================================================================

def excess_count(p, n0, z=3.0, t_max=0.05, n_grid=48):
    """Conservative number of non-null patches in one slide from its patch p-values."""
    p = np.sort(np.asarray(p, float))
    N = len(p)
    ts = np.geomspace(1.0 / (n0 + 1), t_max, n_grid)
    counts = np.searchsorted(p, ts * (1 + 1e-12), "right")
    ex = counts - N * ts - z * np.sqrt(N * ts * (1 - ts))
    return float(max(0.0, ex.max()))


def estimate_n_tumor(scores, null_sorted, z=3.0, t_max=0.05, cap_frac=0.5):
    """scores: patch scores of one tumor slide; null_sorted: ascending normal-patch scores
    of the SAME scorer (never trained on). -> (n_hat >= 1, raw excess)."""
    ex = excess_count(patch_pvalues(null_sorted, scores), len(null_sorted), z, t_max)
    n = int(np.clip(np.round(ex), 1, max(1, int(cap_frac * len(scores)))))
    return n, ex


# =====================================================================================
# 3. One training round
# =====================================================================================

def fit_round(model, bags, labels, pos_idx, epochs=100, lr=1e-3, lr_b=5e-2, patch_weight=1.0,
              neg_weight=0.0, neg_m=32, neg_margin=10.0, init_bias=True):
    """Text-TopK fit + patch-level BCE on pseudo-positives (pos_idx[j] for tumor slide j)
    and on every patch of the normal slides."""
    dev = model.w0.device
    bags = [torch.as_tensor(np.asarray(H) if not torch.is_tensor(H) else H, dtype=torch.float32,
                            device=dev) for H in bags]
    y = torch.as_tensor(np.asarray(labels), dtype=model.w0.dtype, device=dev)
    pos_t = {j: torch.as_tensor(np.asarray(ix), dtype=torch.long, device=dev) for j, ix in pos_idx.items()}
    if init_bias:
        model.init_bias(bags)
    opt = torch.optim.Adam([{"params": model.residual_parameters(), "lr": lr},
                            {"params": [model.bias], "lr": lr_b}])
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        ps = [model.patch_scores(H) for H in bags]
        S = torch.stack([model.pool(s)[0] for s in ps])
        loss = slide_bce_loss(S, y) + model.prior_penalty()
        if neg_weight > 0:
            loss = loss + neg_weight * normal_tail_rank_loss(S, y, ps, neg_m, neg_margin)
        if patch_weight > 0 and pos_t:
            l_pos = torch.stack([F.softplus(-ps[j][ix]).mean() for j, ix in pos_t.items()]).mean()
            l_neg = torch.stack([F.softplus(ps[j]).mean() for j in range(len(bags)) if y[j] == 0]).mean()
            loss = loss + patch_weight * 0.5 * (l_pos + l_neg)
        loss.backward()
        opt.step()
        model.history.append(float(loss.detach()))
    model.eval()
    return model


# =====================================================================================
# 2 + 4. Rounds
# =====================================================================================

@torch.no_grad()
def _scores(fn, H, dev):
    return fn(torch.as_tensor(H, dtype=torch.float32, device=dev)).cpu().numpy().astype(np.float64)


def fit_witness_pseudolabel(bags, labels, make_model, rounds=3, fit_kw=None, patch_weight=1.0,
                            z=3.0, t_max=0.05, cap_frac=0.5, crossfit=True, seed=0, gt=None,
                            ref_k=3, verbose=False):
    """bags: support bags (numpy [N_i, D]); labels 0/1; make_model() -> fresh Text-TopK model
    (TextTopKMIL or MultiConceptTextTopKMIL, on its device). gt: optional {slide index: bool
    mask of tumor patches} for diagnostics. Returns (final model, diagnostics rows)."""
    fit_kw = dict(fit_kw or {})
    labels = np.asarray(labels)
    normals = np.where(labels == 0)[0]
    tumors = np.where(labels == 1)[0]
    rng = np.random.default_rng(seed)
    halves = np.array_split(rng.permutation(normals), 2) if crossfit and len(normals) >= 2 else None
    m0 = make_model()
    dev = m0.w0.device
    # round-0 estimator: the untrained text prior; its null = all support normals
    zs = lambda H: m0.zero_shot_scores(H)                                   # noqa: E731
    estimators = [(zs, np.sort(np.concatenate([_scores(zs, bags[i], dev) for i in normals])))]
    select = zs
    model, rows = None, []
    for r in range(rounds):
        pos_idx = {}
        for j in tumors:
            n_est = []
            for fn, null in estimators:
                n_est.append(estimate_n_tumor(_scores(fn, bags[j], dev), null, z, t_max, cap_frac)[0])
            n_hat = int(round(np.mean(n_est)))
            s_sel = _scores(select, bags[j], dev)
            order = np.argsort(-s_sel)
            pos_idx[j] = order[:n_hat]
            row = {"round": r, "slide": int(j), "n_patches": len(s_sel), "n_hat": n_hat,
                   "pi_hat": n_hat / len(s_sel)}
            if gt is not None and j in gt:
                g = np.asarray(gt[j], bool)
                kk = max(1, min(ref_k, len(s_sel)))
                row.update({"n_true": int(g.sum()), "pi_true": float(g.mean()),
                            "precision": float(g[order[:n_hat]].mean()),
                            "recall": float(g[order[:n_hat]].sum() / max(1, g.sum())),
                            f"precision_top{ref_k}": float(g[order[:kk]].mean())})
            rows.append(row)
        model = fit_round(make_model(), bags, labels, pos_idx, patch_weight=patch_weight, **fit_kw)
        if verbose:
            print(f"  round {r}: n_hat median {np.median([len(v) for v in pos_idx.values()]):.0f}, "
                  f"loss {model.history[-1]:.4f}")
        if r == rounds - 1:
            break
        select = model.patch_scores
        if halves is not None:
            estimators = []
            for held in halves:
                keep = np.setdiff1d(np.arange(len(bags)), held)
                sub_pos = {int(np.where(keep == j)[0][0]): v for j, v in pos_idx.items()}
                fm = fit_round(make_model(), [bags[i] for i in keep], labels[keep], sub_pos,
                               patch_weight=patch_weight, **fit_kw)
                null = np.sort(np.concatenate([_scores(fm.patch_scores, bags[i], dev) for i in held]))
                estimators.append((fm.patch_scores, null))
        else:                                   # no cross-fitting: in-sample null (optimistic)
            estimators = [(model.patch_scores,
                           np.sort(np.concatenate([_scores(model.patch_scores, bags[i], dev) for i in normals])))]
    return model, rows


# =====================================================================================
# Self-test
# =====================================================================================

def self_test():
    import pandas as pd
    from utils.text_topk_mil_torch import TextTopKMIL
    print("self_test: utils.witness_pseudolabel")
    rng = np.random.default_rng(0)

    # --- excess count: null slides ~0, tumor slides ~ number of shifted patches ----------
    n0 = 100_000
    null = np.sort(rng.standard_normal(n0))
    ex_null = [excess_count(patch_pvalues(null, rng.standard_normal(5000)), n0) for _ in range(30)]
    s = rng.standard_normal(5000)
    s[:40] += 6.0
    ex_t = excess_count(patch_pvalues(null, s), n0)
    assert max(ex_null) < 5 and 30 <= ex_t <= 42, (max(ex_null), ex_t)
    s2 = rng.standard_normal(5000)
    s2[:400] += 4.0
    ex_t2 = excess_count(patch_pvalues(null, s2), n0)
    assert 300 <= ex_t2 <= 410, ex_t2
    print(f"  excess count: null slides max {max(ex_null):.1f}; 40 shifted -> {ex_t:.0f}; 400 shifted -> {ex_t2:.0f}")

    # --- end to end on a cohort where top-3 positives are mostly look-alikes -------------
    d = 32
    E = np.eye(d)
    unit = lambda v: v / np.linalg.norm(v)                              # noqa: E731
    w_tum = unit(E[0] + 0.6 * E[1])                                     # tumor
    w_mim = unit(E[0] - 0.6 * E[1])                                     # look-alike (same text score)
    bulk = E[2]
    P = np.column_stack([bulk, unit(E[0])])                             # text knows only "e0 = suspicious"

    def slide(n_t, n=1500, n_mim=6, sig=0.2):
        H = bulk + sig * rng.standard_normal((n, d))
        ii = rng.choice(n, n_mim + n_t, replace=False)
        H[ii[:n_mim]] = w_mim + sig * rng.standard_normal((n_mim, d))
        H[ii[n_mim:]] = w_tum + sig * rng.standard_normal((n_t, d))
        g = np.zeros(n, bool)
        g[ii[n_mim:]] = True
        return (H / np.linalg.norm(H, axis=1, keepdims=True)).astype(np.float32), g

    sizes = [2, 3, 5, 10, 30, 80, 2, 4, 15, 150]
    sup = [slide(0) for _ in range(8)] + [slide(n) for n in sizes[:8]]
    bags = [h for h, _ in sup]
    labels = np.array([0] * 8 + [1] * 8)
    gt = {j: sup[j][1] for j in range(8, 16)}
    evalN = [slide(0)[0] for _ in range(40)]
    evalT = [slide(n)[0] for n in [2, 3, 4] * 10]
    fit_kw = dict(epochs=120, lr=5e-3, neg_weight=1.0, neg_m=16, neg_margin=10.0)
    mk = lambda: TextTopKMIL(P, k=3, lam=0.1)                           # noqa: E731

    def auc(m):
        with torch.no_grad():
            f = lambda H: float(m.slide_score(torch.as_tensor(H))[0])  # noqa: E731
            sn = np.sort([f(H) for H in evalN])
            st = np.array([f(H) for H in evalT])
        return float(np.mean((np.searchsorted(sn, st, "left") + np.searchsorted(sn, st, "right")) / 2 / len(sn)))

    base = mk().fit(bags, labels, **fit_kw)
    model, rows = fit_witness_pseudolabel(bags, labels, mk, rounds=3, fit_kw=fit_kw, patch_weight=1.0,
                                          seed=0, gt=gt)
    diag = pd.DataFrame(rows)
    last = diag[diag["round"] == diag["round"].max()]
    rho = float(pd.Series(last.n_hat.values).corr(pd.Series(last.n_true.values), method="spearman"))
    a0, a1 = auc(base), auc(model)
    print(diag.groupby("round")[["precision", "precision_top3", "recall"]].mean()
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"  last round: Spearman(n_hat, n_true) = {rho:.2f}; low-WR slide AUC Text-TopK {a0:.3f} -> "
          f"pseudo-label {a1:.3f}")
    assert rho > 0.6, rho
    assert last.precision.mean() >= last.precision_top3.mean() - 1e-9
    assert a1 >= a0 - 0.02, (a0, a1)
    assert set(diag["round"]) == {0, 1, 2} and (diag.n_hat >= 1).all()
    print("self_test: OK")


if __name__ == "__main__":
    self_test()
