"""
utils/concept_tail_mil.py

ConceptTail MIL: few-shot slide detection as text-anchored extreme-tail ranking.

Patch scorer (concept-anchored, subspace-constrained), on frozen CONCH embeddings h:
    u_k  = normalize(t_k + B a_k [+ delta_k])          t_k: concept text embedding (unit)
                                                       B [D, r]: orthonormal text subspace
                                                       (SVD of a bank of pathology descriptions)
    z_k  = alpha <h_hat, u_k>                          concept logits
    s(h) = LSE_tau{k in T} z_k - LSE_tau{c in C} z_c + b
           LSE_tau(x) = tau log sum exp(x / tau)       (tau, alpha in logit units, learned)
A patch is tumor when it is closer to SOME tumor concept than to the most similar benign
look-alike. |T| = |C| = 1, a = 0: s = alpha <h_hat, t_T - t_N> + b, i.e. the current Text-TopK
scorer up to the positive constant ||t_T - t_N|| (same ranking).
Parameters: a [r, C] (main), optional delta [D, C] (strong L2), log alpha, log tau, log gamma, b.

Slide pooling (--pool):
    lse   S = gamma log sum_i exp(s_i / gamma)   learned gamma: max as gamma -> 0, then soft
          top-K; never averages a lone tumor patch with normal ones (K <= m).
    topK  S = mean of the K highest s_i            (ablations: top1, top3)

Objective:
    L = BCE(S) + lam_tail L_tail + lam_a sum||a_k||^2 + lam_delta sum||delta_k||^2
        [+ lam_scale (log alpha - log alpha_0)^2]   keeps the learned scale near CONCH's
    L_tail = mean_{i tumor} mean_{j in Q_J} softplus((s_j - S_i + mu) / kappa)
        Q_J = top-J patches of the POOL of every normal support patch, re-mined each step,
        J = ceil(beta N_tot), beta = K_eff / N_eff (guiding proposition: the top of a normal
        slide sits at FPR ~ K/N_eff). N_eff = neff_factor * median patches per normal slide
        (neff_factor < 1 for spatially correlated patches). K_eff = 1 for LSE, K for top-K.
        One-way pAUC surrogate on [0, beta]; the bias cancels (only differences), so the loss
        reshapes the concept prototypes and never moves the threshold.
    neg_weight > 0 adds the earlier per-slide top-M ranking loss (ablation).

API mirrors utils/text_topk_mil_torch.TextTopKMIL (patch_scores, pool, slide_score, forward,
zero_shot_scores, fit, summary, init_bias, residual_parameters, prior_penalty, w0 buffer),
so the analysis scripts and utils/witness_pseudolabel.py accept it.

    python -m utils.concept_tail_mil     # self-test (CPU, < 1 min)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.text_topk_mil_torch import (CONCH_LOGIT_SCALE, load_concepts, normal_tail_rank_loss,
                                       slide_bce_loss)

LOG_CLAMP = (np.log(0.05), np.log(20.0))      # tau, gamma in [0.05, 20] logits


# =====================================================================================
# Text subspace, tail size, losses
# =====================================================================================

def text_subspace(bank=None, r=16, fallback=None):
    """Orthonormal B [D, r'] = top-r left singular vectors of the bank embeddings.
    bank: [n, D] array or an encode_text_bank.py .npz path. Without a bank, fallback
    ([D, C] concept prototypes) is used (r' <= C) -- weaker, printed as a warning."""
    if isinstance(bank, str):
        bank = np.load(bank)["embeddings"]
    if bank is None:
        if fallback is None:
            raise ValueError("need a bank or fallback prototypes")
        print("ConceptTail: no --bank; text subspace = span of the concept prototypes")
        M = np.asarray(fallback, np.float64)
    else:
        M = np.asarray(bank, np.float64).T                     # [D, n]
    M = M / np.linalg.norm(M, axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int((s > 1e-8 * s[0]).sum())
    return U[:, :min(r, rank)].astype(np.float32)


def tail_size(normal_sizes, k_eff=1, beta=None, neff_factor=1.0):
    """(beta, J) with J = ceil(beta * N_tot); beta = k_eff / (neff_factor * median size)."""
    sizes = np.asarray(normal_sizes, float)
    if beta is None:
        beta = k_eff / max(1.0, neff_factor * float(np.median(sizes)))
    return float(beta), int(max(1, np.ceil(beta * sizes.sum())))


def pooled_tail_loss(S, y, patch_scores, J, mu=10.0, kappa=1.0):
    """mean over tumor slides i and the top-J pooled normal patches j of
    softplus((s_j - S_i + mu) / kappa)."""
    y = torch.as_tensor(y, device=S.device).bool()
    if y.all() or (~y).all():
        return S.sum() * 0.0
    neg = torch.cat([ps for ps, yy in zip(patch_scores, y.tolist()) if not yy])
    top = neg.topk(min(int(J), neg.shape[0])).values
    return F.softplus((top[None, :] - S[y][:, None] + mu) / kappa).mean()


# =====================================================================================
# Model
# =====================================================================================

class ConceptTailMIL(nn.Module):
    def __init__(self, concept_protos, is_tumor, B=None, lam_a=0.1, lam_delta=None, pool="lse",
                 k=1, scale=CONCH_LOGIT_SCALE, tau=1.0, gamma=1.0, learn_scale=True, lam_scale=1.0,
                 names=None, dtype=torch.float32):
        super().__init__()
        P = torch.as_tensor(np.asarray(concept_protos), dtype=torch.float64)
        is_tumor = np.asarray(is_tumor, bool)
        if P.shape[1] != len(is_tumor):
            P = P.T
        if P.shape[1] != len(is_tumor):
            raise ValueError(f"protos {tuple(P.shape)} vs {len(is_tumor)} concepts")
        if is_tumor.all() or (~is_tumor).all():
            raise ValueError("need at least one tumor and one normal concept")
        P = P / P.norm(dim=0, keepdim=True)
        D, C = P.shape
        self.register_buffer("protos", P.to(dtype))
        self.register_buffer("tumor_mask", torch.as_tensor(is_tumor))
        w0 = P[:, is_tumor].mean(1) - P[:, ~is_tumor].mean(1)
        self.register_buffer("w0", (w0 / w0.norm()).to(dtype))
        B = np.zeros((D, 0), np.float32) if B is None else np.asarray(B, np.float32)
        if B.shape[0] != D:
            raise ValueError(f"subspace B {B.shape} does not match embedding dim {D}")
        self.register_buffer("B", torch.as_tensor(B, dtype=dtype))
        self.a = nn.Parameter(torch.zeros(B.shape[1], C, dtype=dtype))
        self.lam_a, self.lam_delta, self.lam_scale = lam_a, lam_delta, lam_scale
        self.delta = nn.Parameter(torch.zeros(D, C, dtype=dtype)) if lam_delta is not None else None
        la0 = torch.tensor(float(np.log(scale)), dtype=dtype)
        self.register_buffer("log_alpha0", la0.clone())
        self.register_buffer("log_tau0", torch.tensor(float(np.log(tau)), dtype=dtype))
        self.register_buffer("log_gamma0", torch.tensor(float(np.log(gamma)), dtype=dtype))
        if learn_scale:
            self.log_alpha = nn.Parameter(la0.clone())
        else:
            self.register_buffer("log_alpha", la0.clone())
        self.log_tau = nn.Parameter(self.log_tau0.clone())
        self.pool_rule = pool
        self.log_gamma = nn.Parameter(self.log_gamma0.clone()) if pool == "lse" else None
        self.k = k
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype))
        self.names = list(names) if names is not None else [f"c{i}" for i in range(C)]
        self.history, self.tail_beta, self.tail_J = [], float("nan"), 0

    # ---------------------------------------------------------------- parameters
    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def tau(self):
        return self.log_tau.clamp(*LOG_CLAMP).exp()

    @property
    def gamma(self):
        return self.log_gamma.clamp(*LOG_CLAMP).exp()

    @property
    def k_eff(self):
        return 1 if self.pool_rule == "lse" else self.k

    def prototypes(self):
        U = self.protos + self.B @ self.a
        if self.delta is not None:
            U = U + self.delta
        return U / U.norm(dim=0, keepdim=True)

    def residual_parameters(self):
        return [self.a] + ([self.delta] if self.delta is not None else [])

    def shape_parameters(self):
        out = [self.log_tau] + ([self.log_gamma] if self.log_gamma is not None else [])
        return out + ([self.log_alpha] if isinstance(self.log_alpha, nn.Parameter) else [])

    def prior_penalty(self):
        p = self.lam_a * (self.a ** 2).sum()
        if self.delta is not None:
            p = p + self.lam_delta * (self.delta ** 2).sum()
        if isinstance(self.log_alpha, nn.Parameter):
            p = p + self.lam_scale * (self.log_alpha - self.log_alpha0) ** 2
        return p

    # ---------------------------------------------------------------- scoring
    def concept_logits(self, features, U=None, alpha=None):
        Hn = F.normalize(features.to(self.w0.dtype), dim=-1)
        return (self.alpha if alpha is None else alpha) * (Hn @ (self.prototypes() if U is None else U))

    def _sides(self, z, tau):
        t = tau * torch.logsumexp(z[:, self.tumor_mask] / tau, dim=1)
        n = tau * torch.logsumexp(z[:, ~self.tumor_mask] / tau, dim=1)
        return t, n

    def patch_scores(self, features):
        t, n = self._sides(self.concept_logits(features), self.tau)
        return t - n + self.bias

    @torch.no_grad()
    def zero_shot_scores(self, features):
        """Text prior alone: a = delta = 0, initial alpha and tau, no bias."""
        z = self.concept_logits(features, U=self.protos, alpha=self.log_alpha0.exp())
        t, n = self._sides(z, self.log_tau0.exp())
        return t - n

    def pool(self, s, gamma=None):
        s = torch.as_tensor(s, device=self.w0.device)
        n = s.shape[0]
        if self.pool_rule == "lse":
            g = self.gamma if gamma is None else gamma
            return g * torch.logsumexp(s / g, 0), torch.topk(s, 1).indices
        k = int(np.ceil(self.k * n)) if 0 < self.k < 1 else int(self.k)
        top = torch.topk(s, max(1, min(k, n))).indices
        return s[top].mean(), top

    def slide_score(self, features):
        return self.pool(self.patch_scores(features))

    @torch.no_grad()
    def zero_shot_slide_score(self, features):
        return self.pool(self.zero_shot_scores(features),
                         gamma=self.log_gamma0.exp() if self.pool_rule == "lse" else None)[0]

    def forward(self, features):
        S, top = self.slide_score(features)
        emb = F.normalize(features[top].to(self.w0.dtype).mean(dim=0), dim=-1)
        return torch.stack([torch.zeros_like(S), S]), emb

    @torch.no_grad()
    def explain(self, features, zero_shot=False):
        """Winning tumor concept and winning normal concept per patch (indices into names)."""
        z = self.concept_logits(features, U=self.protos if zero_shot else None)
        ti, ni = torch.where(self.tumor_mask)[0], torch.where(~self.tumor_mask)[0]
        return ti[z[:, ti].argmax(1)], ni[z[:, ni].argmax(1)]

    # ---------------------------------------------------------------- training
    @torch.no_grad()
    def init_bias(self, bags):
        self.bias.zero_()
        self.bias.fill_(-float(np.median([float(self.slide_score(H)[0]) for H in bags])))

    def fit(self, bags, labels, epochs=100, lr=1e-3, lr_b=5e-2, lr_s=1e-2, lam_tail=1.0, tail_beta=None,
            neff_factor=1.0, mu=10.0, kappa=1.0, neg_weight=0.0, neg_m=32, neg_margin=10.0,
            init_bias=True, verbose=False):
        dev = self.w0.device
        bags = [torch.as_tensor(np.asarray(H) if not torch.is_tensor(H) else H, dtype=torch.float32,
                                device=dev) for H in bags]
        y = torch.as_tensor(np.asarray(labels), dtype=self.w0.dtype, device=dev)
        self.tail_beta, self.tail_J = tail_size([len(H) for H, yy in zip(bags, y.tolist()) if yy == 0],
                                                self.k_eff, tail_beta, neff_factor)
        if init_bias:
            self.init_bias(bags)
        groups = [{"params": self.residual_parameters(), "lr": lr}, {"params": [self.bias], "lr": lr_b}]
        if self.shape_parameters():
            groups.append({"params": self.shape_parameters(), "lr": lr_s})
        opt = torch.optim.Adam(groups)
        self.train()
        for ep in range(epochs):
            opt.zero_grad()
            ps = [self.patch_scores(H) for H in bags]
            S = torch.stack([self.pool(s)[0] for s in ps])
            loss = slide_bce_loss(S, y) + self.prior_penalty()
            if lam_tail > 0:
                loss = loss + lam_tail * pooled_tail_loss(S, y, ps, self.tail_J, mu, kappa)
            if neg_weight > 0:
                loss = loss + neg_weight * normal_tail_rank_loss(S, y, ps, neg_m, neg_margin)
            loss.backward()
            opt.step()
            self.history.append(float(loss.detach()))
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d}  loss {float(loss):.4f}")
        self.eval()
        return self

    @torch.no_grad()
    def summary(self):
        U = self.prototypes()
        cos = (U * self.protos).sum(0)
        out = {"alpha": float(self.alpha), "tau": float(self.tau),
               "gamma": float(self.gamma) if self.log_gamma is not None else float("nan"),
               "a_norm": float(self.a.norm()), "delta_norm": float(self.delta.norm()) if self.delta is not None else 0.0,
               "cos_to_text": float(cos.mean()), "min_cos_to_text": float(cos.min()),
               "bias": float(self.bias), "tail_beta": self.tail_beta, "tail_J": self.tail_J,
               "rank_B": int(self.B.shape[1]), "n_concepts": int(self.protos.shape[1]),
               "final_loss": self.history[-1] if self.history else float("nan")}
        for n, v in zip(self.names, cos.tolist()):
            out[f"cos_to_text[{n}]"] = v
        return out


def make_concept_tail(concepts, bank=None, r=16, **kw):
    """concepts: encode_concepts.py .npz path or (protos, names, is_tumor); bank: .npz path,
    [n, D] array or None (subspace of the concepts themselves)."""
    protos, names, is_tumor = load_concepts(concepts) if isinstance(concepts, str) else concepts
    P = np.asarray(protos)
    if P.shape[1] != len(is_tumor):
        P = P.T
    B = text_subspace(bank, r, fallback=P) if r > 0 else None
    return ConceptTailMIL(P, is_tumor, B=B, names=names, **kw)


# =====================================================================================
# Self-test
# =====================================================================================

def self_test():
    from utils.text_topk_mil_torch import TextTopKMIL
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    print("self_test: utils.concept_tail_mil")
    unit = lambda v: v / np.linalg.norm(v)                              # noqa: E731

    # 1. reduction to Text-TopK with one concept per side and a = 0
    d = 16
    tN, tT = unit(rng.standard_normal(d)), unit(rng.standard_normal(d))
    H = torch.as_tensor(rng.standard_normal((50, d)), dtype=torch.float32)
    m = ConceptTailMIL(np.column_stack([tT, tN]), [True, False], B=text_subspace(rng.standard_normal((20, d)), 4),
                       pool="top3", k=3)
    ref = TextTopKMIL(np.column_stack([tN, tT]), k=3)
    c = float(np.linalg.norm(tT - tN))
    assert torch.allclose(m.patch_scores(H), c * ref.patch_scores(H), atol=1e-4)
    assert torch.allclose(m.zero_shot_scores(H), c * ref.zero_shot_scores(H), atol=1e-4)
    print(f"  1+1 concepts, a = 0: scores = {c:.3f} x Text-TopK scores")

    # 2. gradients (double precision), LSE pooling limits, bias invariance, tail size
    md = ConceptTailMIL(rng.standard_normal((d, 4)), [True, True, False, False],
                        B=text_subspace(rng.standard_normal((20, d)), 5), lam_delta=1.0, dtype=torch.float64)
    with torch.no_grad():
        md.a.normal_(0, 0.1)
        md.delta.normal_(0, 0.05)
    Hd = torch.as_tensor(rng.standard_normal((30, d)), dtype=torch.float64)
    from torch.func import functional_call
    for name in ("a", "delta", "log_tau", "log_gamma", "log_alpha"):
        p0 = getattr(md, name).detach().clone().requires_grad_()
        torch.autograd.gradcheck(lambda p, n=name: functional_call(md, {n: p}, (Hd,))[0][1], (p0,))
    params = [p for p in md.parameters() if p.requires_grad]
    loss = md.slide_score(Hd)[0] + md.prior_penalty()
    g = torch.autograd.grad(loss, params, allow_unused=True)
    assert all(gi is not None and torch.isfinite(gi).all() for gi in g)
    s = torch.tensor([3.0, 1.0, -2.0, 0.5])
    assert abs(float(md.pool(s, gamma=torch.tensor(0.01))[0]) - 3.0) < 1e-3
    assert float(md.pool(s, gamma=torch.tensor(5.0))[0]) > 3.0
    ps = [torch.randn(200), torch.randn(150), torch.randn(100) + 1]
    S = torch.stack([p.max() for p in ps])
    y = torch.tensor([0, 0, 1])
    l1 = pooled_tail_loss(S, y, ps, 10)
    l2 = pooled_tail_loss(S + 7, y, [p + 7 for p in ps], 10)
    assert torch.allclose(l1, l2)
    beta, J = tail_size([1000, 3000, 2000], k_eff=1)
    assert abs(beta - 1 / 2000) < 1e-12 and J == 3
    assert tail_size([1000, 3000, 2000], 3, neff_factor=0.25)[1] == 36
    print(f"  gradcheck OK; LSE pool -> max as gamma -> 0; tail loss bias-invariant; J(beta=1/2000) = {J}")

    # 3. mechanism: benign look-alikes share the text "tumor" direction ---------------------
    d = 32
    E = np.eye(d)
    tum = unit(E[0] + 0.7 * E[1])                   # tumor
    mim = unit(E[0] - 0.7 * E[1])                   # benign look-alike (e.g. sinus histiocytes)
    bulk = E[2]                                     # lymphocytes

    def slide(n_t, n=1200, n_mim=10, sig=0.25):
        X = bulk + sig * rng.standard_normal((n, d))
        ii = rng.choice(n, n_mim + n_t, replace=False)
        X[ii[:n_mim]] = mim + sig * rng.standard_normal((n_mim, d))
        X[ii[n_mim:]] = tum + sig * rng.standard_normal((n_t, d))
        return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)

    # text knows: tumor ~ e0 ("atypical cells"), normal ~ bulk, look-alike concept imperfect
    tT = unit(E[0] + 0.2 * rng.standard_normal(d))
    tL = unit(mim + 0.6 * unit(rng.standard_normal(d)))
    concepts = np.column_stack([tT, bulk, tL])
    bank = np.stack([unit(E[i] + 0.1 * rng.standard_normal(d)) for i in range(6)]
                    + [unit(rng.standard_normal(d)) for _ in range(20)])
    sup = [slide(0) for _ in range(8)] + [slide(n) for n in (1, 2, 3, 5, 10, 30, 2, 60)]
    yl = np.array([0] * 8 + [1] * 8)
    evN = [slide(0) for _ in range(40)]
    evT = [slide(n) for n in [1, 2, 3] * 10]

    def auc(score):
        sn = np.sort([score(X) for X in evN])
        st = np.array([score(X) for X in evT])
        return float(np.mean((np.searchsorted(sn, st, "left") + np.searchsorted(sn, st, "right")) / 2 / len(sn)))

    single = TextTopKMIL(np.column_stack([bulk, tT]), k=1)
    zs1 = auc(lambda X: float(single.zero_shot_scores(torch.as_tensor(X)).max()))
    mk = lambda **kw: ConceptTailMIL(concepts, [True, False, False], B=text_subspace(bank, 8), **kw)  # noqa: E731
    zs = mk()
    zsc = auc(lambda X: float(zs.zero_shot_slide_score(torch.as_tensor(X))))
    fit_kw = dict(epochs=150, lr=1e-2, lam_tail=1.0, neff_factor=0.5)

    def trained(model):
        model.fit(sup, yl, **fit_kw)
        return auc(lambda X: float(model.slide_score(torch.as_tensor(X))[0].detach())), model

    tt = TextTopKMIL(np.column_stack([bulk, tT]), k=3, lam=0.1).fit(sup, yl, epochs=150, lr=1e-2,
                                                                    neg_weight=1.0, neg_m=32, neg_margin=10)
    a_tt = auc(lambda X: float(tt.slide_score(torch.as_tensor(X))[0].detach()))
    a_ct, m_ct = trained(mk())
    a_ct0, _ = trained(ConceptTailMIL(concepts, [True, False, False], B=None))       # no subspace
    print(f"  low-WR slide AUC (1-3 tumor patches, 10 look-alikes/slide): zero-shot single direction "
          f"{zs1:.3f}, zero-shot concepts {zsc:.3f}; trained Text-TopK(K=3, tail-rank) {a_tt:.3f}, "
          f"ConceptTail r=0 {a_ct0:.3f}, ConceptTail r=8 {a_ct:.3f}")
    sm = m_ct.summary()
    print(f"  ConceptTail r=8: alpha {sm['alpha']:.1f}, tau {sm['tau']:.2f}, gamma {sm['gamma']:.2f}, "
          f"||a|| {sm['a_norm']:.3f}, J {sm['tail_J']}, cos to text {sm['cos_to_text']:.3f}")
    assert zsc > zs1 + 0.1, (zs1, zsc)
    assert a_ct >= a_ct0 - 0.02 and a_ct > a_tt - 0.02, (a_ct, a_ct0, a_tt)
    # explain(): look-alike patches are won by the look-alike concept
    _, win_n = m_ct.explain(torch.as_tensor(np.stack([mim, bulk]), dtype=torch.float32), zero_shot=True)
    assert win_n.tolist() == [2, 1], win_n
    X = torch.as_tensor(evN[0])
    # forward() convention and the witness pseudo-labelling hooks
    logits, emb = m_ct(X)
    assert logits.shape == (2,) and float(logits[0]) == 0.0 and emb.shape == (d,)
    from utils.witness_pseudolabel import fit_witness_pseudolabel
    pm, diag = fit_witness_pseudolabel(sup, yl, mk, rounds=2, fit_kw={"epochs": 40, "lr": 1e-2}, seed=0)
    assert len(diag) == 16
    print("self_test: OK")


if __name__ == "__main__":
    self_test()
