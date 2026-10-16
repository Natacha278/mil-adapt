"""
utils/text_topk_mil.py

Text-initialised top-K MIL on frozen patch embeddings (steps 1-3 of the method):

  1. patch scorer   s(h) = alpha * <h, w0 + Delta> + b
                    w0 = (t_Tumor - t_Normal) / ||t_Tumor - t_Normal||  (text prior, fixed)
                    Delta: learned residual (D parameters), b: learned bias,
                    alpha: fixed temperature (CONCH logit scale), so at Delta = 0, b = 0 the
                    score is exactly the zero-shot logit margin.
  2. slide score    S_j = mean of the K highest s(h) in slide j  (K < 1: fraction of patches)
  3. objective      L = mean_j BCE(sigmoid(S_j), y_j) + lam * ||Delta||^2

  Gradient (analytic, linear model; top-K membership re-evaluated every step):
      dL/dS_j     = (sigmoid(S_j) - y_j) / n_slides
      dL/dDelta   = sum_j dL/dS_j * alpha * mean_{i in topK(j)} h_ij + 2 lam Delta
      dL/db       = sum_j dL/dS_j
  Normal slides push the direction away from their most tumor-like patches (hard
  negatives with certain labels); tumor slides pull it toward their current top-K
  patches, which the text initialisation makes mostly real lesion patches.

Pure numpy: no GPU, no torch. Full-batch Adam over the support slides.
Prediction: Tumor iff S_j > 0 (the threshold of steps 1-3; step 4 would replace it).

    python -m utils.text_topk_mil      # self-test: gradient check + synthetic benchmark
"""

import numpy as np

CONCH_LOGIT_SCALE = 56.347694396972656


def _unit_rows(H):
    H = np.asarray(H, np.float32)
    n = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.maximum(n, 1e-12)


def text_direction(text_prototypes):
    """w0 from a [D, 2] (or [2, D]) prototype array with columns [Normal, Tumor]."""
    P = np.asarray(text_prototypes, np.float64)
    if P.shape[0] < P.shape[1]:
        P = P.T
    P = P / np.linalg.norm(P, axis=0, keepdims=True)
    w0 = P[:, 1] - P[:, 0]
    return w0 / np.linalg.norm(w0)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, float)))     # overflow-free


class TextTopKMIL:
    def __init__(self, text_prototypes=None, w0=None, k=1, lam=0.1,
                 scale=CONCH_LOGIT_SCALE):
        w0 = text_direction(text_prototypes) if w0 is None else np.asarray(w0, float)
        self.w0 = w0 / np.linalg.norm(w0)
        self.D = np.zeros_like(self.w0)
        self.b = 0.0
        self.k, self.lam, self.scale = k, lam, scale
        self.history = []

    # ---------------------------------------------------------------- scoring
    def _k(self, n):
        return int(np.clip(np.ceil(self.k * n) if 0 < self.k < 1 else self.k, 1, n))

    def patch_scores(self, H):
        return self.scale * (_unit_rows(H) @ (self.w0 + self.D)) + self.b

    def slide_score(self, H, return_top=False):
        Hn = _unit_rows(H)
        s = self.scale * (Hn @ (self.w0 + self.D)) + self.b
        k = self._k(len(s))
        top = np.argpartition(-s, k - 1)[:k]
        S = float(s[top].mean())
        return (S, Hn[top].mean(axis=0)) if return_top else S

    def predict_proba(self, bags):
        return np.array([sigmoid(self.slide_score(H)) for H in bags])

    # ---------------------------------------------------------------- training
    def loss_and_grad(self, bags, y):
        n = len(bags)
        loss, gD, gb = 0.0, np.zeros_like(self.D), 0.0
        for H, yj in zip(bags, y):
            S, hbar = self.slide_score(H, return_top=True)
            p = sigmoid(S)
            loss += -(yj * np.log(max(p, 1e-12)) + (1 - yj) * np.log(max(1 - p, 1e-12))) / n
            g = (p - yj) / n
            gD += g * self.scale * hbar
            gb += g
        loss += self.lam * float(self.D @ self.D)
        gD += 2 * self.lam * self.D
        return loss, gD, gb

    def init_bias(self, bags):
        """Centre the initial slide scores so training starts from S around 0."""
        self.b = 0.0
        self.b = -float(np.median([self.slide_score(H) for H in bags]))

    def fit(self, bags, y, epochs=100, lr=1e-3, lr_b=5e-2, betas=(0.9, 0.999), eps=1e-8,
            init_bias=True):
        y = np.asarray(y, float)
        if init_bias:
            self.init_bias(bags)
        mD, vD = np.zeros_like(self.D), np.zeros_like(self.D)
        mb = vb = 0.0
        b1, b2 = betas
        for t in range(1, epochs + 1):
            loss, gD, gb = self.loss_and_grad(bags, y)
            self.history.append(loss)
            mD = b1 * mD + (1 - b1) * gD; vD = b2 * vD + (1 - b2) * gD ** 2
            mb = b1 * mb + (1 - b1) * gb; vb = b2 * vb + (1 - b2) * gb ** 2
            c1, c2 = 1 - b1 ** t, 1 - b2 ** t
            self.D -= lr * (mD / c1) / (np.sqrt(vD / c2) + eps)
            self.b -= lr_b * (mb / c1) / (np.sqrt(vb / c2) + eps)
        return self

    def summary(self):
        w = self.w0 + self.D
        cos = float(w @ self.w0 / np.linalg.norm(w))
        assert -1 - 1e-9 <= cos <= 1 + 1e-9
        return {"delta_norm": float(np.linalg.norm(self.D)),
                "cos_to_text": cos,
                "bias": float(self.b),
                "final_loss": float(self.history[-1]) if self.history else np.nan}


# =====================================================================================
# Self-test
# =====================================================================================

def _auc(pos, neg):
    neg = np.sort(neg)
    lt = np.searchsorted(neg, pos, "left"); le = np.searchsorted(neg, pos, "right")
    return float(((lt + 0.5 * (le - lt)) / len(neg)).mean())


def self_test():
    print("self_test: utils.text_topk_mil")
    rng = np.random.default_rng(0)
    d = 32

    # --- 1. analytic gradient == finite differences ---------------------------------
    m = TextTopKMIL(w0=np.eye(d)[0], k=3, lam=0.3, scale=20.0)
    m.D = 0.05 * rng.standard_normal(d); m.b = 0.2
    bags = [rng.standard_normal((50, d)) for _ in range(6)]
    y = np.array([0, 1, 0, 1, 1, 0], float)
    _, gD, gb = m.loss_and_grad(bags, y)
    num = np.zeros(d)
    for i in range(d):
        for sgn in (1, -1):
            m.D[i] += sgn * 1e-6
            num[i] += sgn * m.loss_and_grad(bags, y)[0] / 2e-6
            m.D[i] -= sgn * 1e-6
    m.b += 1e-6; lp = m.loss_and_grad(bags, y)[0]; m.b -= 2e-6
    lm = m.loss_and_grad(bags, y)[0]; m.b += 1e-6
    assert np.allclose(gD, num, atol=1e-5), np.abs(gD - num).max()
    assert abs(gb - (lp - lm) / 2e-6) < 1e-5
    print(f"  analytic gradient matches finite differences (max |diff| {np.abs(gD - num).max():.1e})")

    # --- 2. the failure mode the method targets ---------------------------------------
    # Text direction e0 sees both tumor and benign mimics; tumor also has e1, mimics -e1.
    # Every normal slide has a few mimics, so zero-shot top-K is fooled at low witness rate.
    def make_slide(tumor, n=2000, wr=0.002):
        H = rng.standard_normal((n, d)) * 0.3
        mimic = rng.choice(n, 6, replace=False)
        H[mimic, 0] += 1.4; H[mimic, 1] -= 1.0
        if tumor:
            k = max(1, int(round(wr * n)))
            idx = rng.choice(np.setdiff1d(np.arange(n), mimic), k, replace=False)
            H[idx, 0] += 1.2; H[idx, 1] += 1.0
        return H

    w0 = np.eye(d)[0] + 0.15 * rng.standard_normal(d)
    tr_bags = [make_slide(t, wr=rng.choice([0.002, 0.01, 0.05])) for t in [0] * 16 + [1] * 16]
    tr_y = np.array([0] * 16 + [1] * 16)
    te_norm = [make_slide(False) for _ in range(60)]
    te_tum = {wr: [make_slide(True, wr=wr) for _ in range(40)] for wr in (0.001, 0.005, 0.05)}

    zs = TextTopKMIL(w0=w0, k=1, lam=0.1, scale=20.0)
    zs.init_bias(tr_bags)                                 # same bias rule, no training
    mdl = TextTopKMIL(w0=w0, k=1, lam=0.1, scale=20.0).fit(tr_bags, tr_y, epochs=150, lr=5e-3)
    print(f"  trained: {mdl.summary()}")
    sn_z = [zs.slide_score(H) for H in te_norm]
    sn_m = [mdl.slide_score(H) for H in te_norm]
    for wr, bags_t in te_tum.items():
        az = _auc(np.array([zs.slide_score(H) for H in bags_t]), np.array(sn_z))
        am = _auc(np.array([mdl.slide_score(H) for H in bags_t]), np.array(sn_m))
        acc_m = np.mean([mdl.slide_score(H) > 0 for H in bags_t])
        print(f"  witness rate {wr:.3f}: slide AUC zero-shot {az:.3f} -> trained {am:.3f}; "
              f"tumor slides detected {acc_m:.2f}")
        if wr == 0.001:
            assert am > az + 0.15, (az, am)
    spec = np.mean([s <= 0 for s in sn_m])
    print(f"  normal slides kept Normal: {spec:.2f}")
    assert mdl.history[-1] < mdl.history[0]
    print("self_test: OK")


if __name__ == "__main__":
    self_test()
