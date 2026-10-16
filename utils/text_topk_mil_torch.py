"""
utils/text_topk_mil_torch.py

PyTorch version of Text-TopK MIL (utils/text_topk_mil.py is the numpy reference; with
the same settings both give the same parameters -- checked by the self-test).

Model (steps 1-3)
    s(h) = alpha * <h_hat, w0 + Delta> + b                       patch score
           [ - gamma * alpha * max_m <h_hat, t_m> ]               optional benign-mimic term
    S    = mean of the K highest s(h) in the slide                slide score
    logits = [0, S]  ->  softmax(logits)[Tumor] = sigmoid(S)     (repo adapter convention)
    L    = mean BCE(sigmoid(S_j), y_j) + lam * ||Delta||^2  [+ extra_loss(...)]

Compatible with the rest of the repo:
  * forward(features) -> (logits[2], embedding), like utils/adapters.py, so
    utils.trainer.validate_model and the analysis scripts work unchanged; prediction is
    argmax = Tumor iff S > 0.
  * model.fit(bags, labels, ...) is the training loop to use: full-batch Adam with the
    prior penalty (utils.trainer.train_model would run per-slide AdamW without it).

Multi-concept scorer (MultiConceptTextTopKMIL, make_text_topk(concepts=...))
    Several text concepts per side, each with its own residual; patch score =
    smooth-max over tumor concepts - smooth-max over normal concepts + b. Concept files
    come from encode_concepts.py. Same fit / ranking loss / conformal wrapper.

Normal-tail ranking loss (neg_weight > 0 in fit)
    Every patch of a normal slide is a certain negative, but the slide BCE with K=1 only
    sends gradient through ONE patch per normal slide (16 negatives at 16 shots). This
    loss uses the top-M patches of EVERY normal support slide, pooled into a set N of
    hard negatives (re-mined each step), and asks each tumor slide's score to beat them:
        L_rank = mean_{j tumor} mean_{i in N} softplus(margin + s_i - S_j)
    A pairwise surrogate of the partial AUC at low false-positive rate -- the operating
    region a high-specificity (conformal) threshold uses. The bias b cancels in s_i - S_j,
    so the loss moves only the direction (Delta), not the cut.
        L = BCE + lam ||Delta||^2 + neg_weight * L_rank

Hooks for the next steps
  * extra_loss=callable(S, y, model) added to the objective, e.g.
        lambda S, y, m: 0.5 * tail_quantile_loss(S, y, q=0.95, margin=1.0)
    (the differentiable form of step 4: every tumor slide above the q-quantile of the
    normal slides' scores; the gradient reaches the normal slide(s) at that quantile).
  * mimic_embeddings=[D, M] text prototypes of benign look-alikes (germinal centre,
    sinus histiocytes, ...): subtracts gamma * alpha * max_m cos(h, t_m) from each patch
    score; gamma fixed (default) or learned (learn_gamma=True).
  * patch_scores() / slide_score() are separate so other pooling rules can be swapped in.

    python -m utils.text_topk_mil_torch     # self-test (CPU): gradient check, numpy parity, hooks
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CONCH_LOGIT_SCALE = 56.347694396972656
TUMOR = 1


def _text_direction(text_embeddings):
    P = torch.as_tensor(np.asarray(text_embeddings), dtype=torch.float64)
    if P.shape[0] < P.shape[1]:
        P = P.T
    P = P / P.norm(dim=0, keepdim=True)
    w0 = P[:, TUMOR] - P[:, 1 - TUMOR]
    return w0 / w0.norm()


def normal_tail_rank_loss(S, y, patch_scores, m=64, margin=1.0):
    """S: [n_slides] slide scores; y: [n_slides] labels; patch_scores: list of [n_i]
    patch-score tensors (same order). Hard negatives = top-m patches of every normal
    slide, pooled. Returns mean softplus(margin + s_neg - S_tumor) over all pairs."""
    y = torch.as_tensor(y).bool()
    if y.all() or (~y).all():
        return S.sum() * 0.0
    neg = torch.cat([ps.topk(min(m, ps.shape[0])).values
                     for ps, yy in zip(patch_scores, y.tolist()) if not yy])
    return F.softplus(margin + neg[None, :] - S[y][:, None]).mean()


def slide_bce_loss(S, y):
    """Mean binary cross-entropy of sigmoid(S) against slide labels (stable form)."""
    return F.binary_cross_entropy_with_logits(S, y.to(S.dtype))


def tail_quantile_loss(S, y, q=0.95, margin=1.0):
    """softplus(q-quantile of normal-slide scores + margin - S_tumor), mean over tumor
    slides. Differentiable (torch.quantile interpolates between order statistics)."""
    y = y.bool()
    if y.all() or (~y).all():
        return S.sum() * 0.0
    thr = torch.quantile(S[~y], q)
    return F.softplus(thr + margin - S[y]).mean()


class TextTopKMIL(nn.Module):
    def __init__(self, text_embeddings, k=1, lam=0.1, scale=CONCH_LOGIT_SCALE,
                 mimic_embeddings=None, gamma=0.0, learn_gamma=False, dtype=torch.float32):
        super().__init__()
        w0 = _text_direction(text_embeddings).to(dtype)
        self.register_buffer("w0", w0)
        self.delta = nn.Parameter(torch.zeros_like(w0))
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype))
        self.k, self.lam, self.scale = k, lam, scale
        if mimic_embeddings is not None:
            M = torch.as_tensor(np.asarray(mimic_embeddings), dtype=torch.float64)
            if M.shape[0] != w0.shape[0]:
                M = M.T
            self.register_buffer("mimics", (M / M.norm(dim=0, keepdim=True)).to(dtype))
            g = torch.tensor(float(gamma), dtype=dtype)
            if learn_gamma:
                self.gamma = nn.Parameter(g)
            else:
                self.register_buffer("gamma", g)
        else:
            self.mimics = None
        self.history = []

    # ---------------------------------------------------------------- scoring
    def _k(self, n):
        k = int(np.ceil(self.k * n)) if 0 < self.k < 1 else int(self.k)
        return max(1, min(k, n))

    def patch_scores(self, features):
        Hn = F.normalize(features.to(self.w0.dtype), dim=-1)
        s = self.scale * (Hn @ (self.w0 + self.delta)) + self.bias
        if self.mimics is not None:
            s = s - self.gamma * self.scale * (Hn @ self.mimics).max(dim=1).values
        return s

    def pool(self, s):
        """Top-K mean of patch scores s -> (S, top indices)."""
        top = torch.topk(s, self._k(s.shape[0])).indices
        return s[top].mean(), top

    def slide_score(self, features):
        return self.pool(self.patch_scores(features))

    def forward(self, features):
        S, top = self.slide_score(features)
        logits = torch.stack([torch.zeros_like(S), S])
        embedding = F.normalize(features[top].to(self.w0.dtype).mean(dim=0), dim=-1)
        return logits, embedding

    def prior_penalty(self):
        return self.lam * (self.delta ** 2).sum()

    def residual_parameters(self):
        return [self.delta]

    @torch.no_grad()
    def zero_shot_scores(self, features):
        """Patch scores of the text prior alone (Delta = 0, b = 0)."""
        Hn = F.normalize(features.to(self.w0.dtype), dim=-1)
        return self.scale * (Hn @ self.w0)

    # ---------------------------------------------------------------- training
    @torch.no_grad()
    def init_bias(self, bags):
        """Centre the initial slide scores (numpy median, i.e. the mean of the two middle
        values for an even count -- same rule as the numpy reference)."""
        self.bias.zero_()
        S = [float(self.slide_score(H)[0]) for H in bags]
        self.bias.fill_(-float(np.median(S)))

    def fit(self, bags, labels, epochs=100, lr=1e-3, lr_b=5e-2, betas=(0.9, 0.999),
            eps=1e-8, init_bias=True, extra_loss=None, neg_weight=0.0, neg_m=64,
            neg_margin=1.0, verbose=False):
        """neg_weight > 0 adds the normal-tail ranking loss (top-neg_m patches of every
        normal slide as hard negatives; see the module docstring)."""
        dev = self.w0.device
        bags = [torch.as_tensor(np.asarray(H) if not torch.is_tensor(H) else H,
                                dtype=torch.float32, device=dev) for H in bags]
        y = torch.as_tensor(np.asarray(labels), dtype=self.w0.dtype, device=dev)
        if init_bias:
            self.init_bias(bags)
        groups = [{"params": self.residual_parameters(), "lr": lr}, {"params": [self.bias], "lr": lr_b}]
        if self.mimics is not None and isinstance(self.gamma, nn.Parameter):
            groups.append({"params": [self.gamma], "lr": lr_b})
        opt = torch.optim.Adam(groups, betas=betas, eps=eps)
        self.train()
        for ep in range(epochs):
            opt.zero_grad()
            ps = [self.patch_scores(H) for H in bags]
            S = torch.stack([self.pool(s)[0] for s in ps])
            loss = slide_bce_loss(S, y) + self.prior_penalty()
            if neg_weight > 0:
                loss = loss + neg_weight * normal_tail_rank_loss(S, y, ps, neg_m, neg_margin)
            if extra_loss is not None:
                loss = loss + extra_loss(S, y, self)
            loss.backward()
            opt.step()
            self.history.append(float(loss.detach()))
            if verbose and (ep % 20 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:4d}  loss {float(loss):.4f}")
        self.eval()
        return self

    @torch.no_grad()
    def summary(self):
        w = self.w0 + self.delta
        out = {"delta_norm": float(self.delta.norm()),
               "cos_to_text": float(w @ self.w0 / w.norm()),
               "bias": float(self.bias),
               "final_loss": self.history[-1] if self.history else float("nan")}
        if self.mimics is not None:
            out["gamma"] = float(self.gamma)
        return out


def load_concepts(path):
    """Concept file written by encode_concepts.py -> (protos [D, C], names, is_tumor [C])."""
    z = np.load(path, allow_pickle=False)
    return z["protos"], [str(n) for n in z["names"]], z["is_tumor"].astype(bool)


class MultiConceptTextTopKMIL(TextTopKMIL):
    """Text-TopK with several text concepts per side (normal tail modelled by exclusion).

    Each concept c has a fixed text prototype t_c (unit norm) and a learned residual d_c:
        z_c(h) = alpha * <h_hat, t_c + d_c>
        T(h)   = tau * logsumexp_{c tumor}(z_c / tau)       smooth max over tumor concepts
        N(h)   = tau * logsumexp_{c normal}(z_c / tau)      smooth max over normal concepts
        s(h)   = T(h) - N(h) + b
    A patch scores high only if it looks like SOME tumor concept and unlike EVERY normal
    concept. With one concept per side this is the current Text-TopK up to a positive
    constant (||t_T - t_N||), i.e. the same ranking and the same decisions.

    Optional group residuals (lam_leaf given): d_c = D_side(c) + e_c with
        prior lam * ||D_side||^2 + lam_leaf * ||e_c||^2  (lam_leaf > lam: concepts borrow
    strength from their side -- hierarchical shrinkage for few-shot).
    Without lam_leaf: independent residuals, prior lam * sum_c ||d_c||^2.

    Everything else (top-K pooling, fit with BCE + normal-tail ranking, conformal wrapper,
    forward() convention) is inherited unchanged: the ranking loss's gradient reaches the
    normal concept that wins the smooth max for each hard negative.
    """

    def __init__(self, concept_protos, is_tumor, k=1, lam=0.1, scale=CONCH_LOGIT_SCALE,
                 tau=1.0, lam_leaf=None, names=None, dtype=torch.float32):
        nn.Module.__init__(self)
        P = torch.as_tensor(np.asarray(concept_protos), dtype=torch.float64)
        is_tumor = np.asarray(is_tumor, bool)
        if P.shape[1] != len(is_tumor):
            P = P.T
        if P.shape[1] != len(is_tumor):
            raise ValueError(f"protos {tuple(P.shape)} vs {len(is_tumor)} concepts")
        if is_tumor.all() or (~is_tumor).all():
            raise ValueError("need at least one tumor and one normal concept")
        P = P / P.norm(dim=0, keepdim=True)
        self.register_buffer("protos", P.to(dtype))
        self.register_buffer("tumor_mask", torch.as_tensor(is_tumor))
        w0 = P[:, is_tumor].mean(1) - P[:, ~is_tumor].mean(1)        # reference direction
        self.register_buffer("w0", (w0 / w0.norm()).to(dtype))
        C, D = P.shape[1], P.shape[0]
        self.delta = nn.Parameter(torch.zeros(D, C, dtype=dtype))   # per-concept residuals
        self.lam_leaf = lam_leaf
        if lam_leaf is not None:
            self.delta_side = nn.Parameter(torch.zeros(D, 2, dtype=dtype))   # [normal, tumor]
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype))
        self.k, self.lam, self.scale, self.tau = k, lam, scale, tau
        self.names = list(names) if names is not None else [f"c{i}" for i in range(C)]
        self.mimics = None
        self.history = []

    # ---------------------------------------------------------------- scoring
    def effective_protos(self):
        W = self.protos + self.delta
        if self.lam_leaf is not None:
            W = W + self.delta_side[:, self.tumor_mask.long()]
        return W

    def concept_scores(self, features, W=None):
        Hn = F.normalize(features.to(self.w0.dtype), dim=-1)
        return self.scale * (Hn @ (self.effective_protos() if W is None else W))

    def _side_scores(self, z):
        t = self.tau * torch.logsumexp(z[:, self.tumor_mask] / self.tau, dim=1)
        n = self.tau * torch.logsumexp(z[:, ~self.tumor_mask] / self.tau, dim=1)
        return t, n

    def patch_scores(self, features):
        t, n = self._side_scores(self.concept_scores(features))
        return t - n + self.bias

    @torch.no_grad()
    def zero_shot_scores(self, features):
        t, n = self._side_scores(self.concept_scores(features, W=self.protos))
        return t - n

    @torch.no_grad()
    def explain(self, features, zero_shot=False):
        """Winning tumor concept and winning normal concept per patch (indices into names)."""
        z = self.concept_scores(features, W=self.protos if zero_shot else None)
        ti = torch.where(self.tumor_mask)[0]
        ni = torch.where(~self.tumor_mask)[0]
        return ti[z[:, ti].argmax(1)], ni[z[:, ni].argmax(1)]

    def prior_penalty(self):
        if self.lam_leaf is None:
            return self.lam * (self.delta ** 2).sum()
        return self.lam * (self.delta_side ** 2).sum() + self.lam_leaf * (self.delta ** 2).sum()

    def residual_parameters(self):
        return [self.delta] + ([self.delta_side] if self.lam_leaf is not None else [])

    @torch.no_grad()
    def summary(self):
        W = self.effective_protos()
        cos = (W * self.protos).sum(0) / W.norm(dim=0)
        d = W - self.protos
        out = {"delta_norm": float(d.norm()),
               "cos_to_text": float(cos.mean()),
               "bias": float(self.bias),
               "final_loss": self.history[-1] if self.history else float("nan"),
               "n_concepts": int(self.protos.shape[1]),
               "n_tumor_concepts": int(self.tumor_mask.sum())}
        for n, v in zip(self.names, d.norm(dim=0).tolist()):
            out[f"delta_norm[{n}]"] = v
        return out


def make_text_topk(text_prototypes=None, concepts=None, k=1, lam=0.1, tau=1.0, lam_leaf=None,
                   dtype=torch.float32):
    """Two-prototype Text-TopK (text_prototypes [D, 2]) or the multi-concept model
    (concepts = path of an encode_concepts.py file, or (protos, names, is_tumor))."""
    if concepts is None:
        return TextTopKMIL(text_prototypes, k=k, lam=lam, dtype=dtype)
    protos, names, is_tumor = load_concepts(concepts) if isinstance(concepts, str) else concepts
    return MultiConceptTextTopKMIL(protos, is_tumor, k=k, lam=lam, tau=tau, lam_leaf=lam_leaf,
                                   names=names, dtype=dtype)


class ConformalTextTopK(nn.Module):
    """Step 4: Text-TopK with the leave-one-normal-out conformal threshold
    (utils/conformal.py). Holds one TextTopKMIL per support normal, each trained without
    that normal, and the normal's held-out score z_j.

    forward(features) -> (logits[2], embedding) like every adapter, with
        logits = [0, c_max + 0.5 - count(x)]
    so argmax = Tumor iff count(x) <= c_max iff p(x) <= alpha (no ties possible), and
    utils.trainer.validate_model works unchanged. The logit is a decision value only;
    use evaluate()["margin"] as a continuous slide score.
    """

    def __init__(self, models, z, alpha=0.1):
        super().__init__()
        from utils.conformal import count_max
        self._count_max = count_max
        self.members = nn.ModuleList(models)
        self.register_buffer("z", torch.as_tensor(np.asarray(z), dtype=models[0].w0.dtype))
        self.set_alpha(alpha)

    def set_alpha(self, alpha):
        self.alpha, self.c_max = alpha, self._count_max(len(self.z), alpha)
        return self

    @classmethod
    def fit_loo(cls, bags, labels, make_model, alpha=0.1, verbose=False, **fit_kw):
        """make_model() -> fresh TextTopKMIL (already on its device); fit_kw go to .fit()."""
        from utils.conformal import count_max, loo_normal_fits
        count_max(int((np.asarray(labels) == 0).sum()), alpha)        # fail before training

        def score(m, H):
            with torch.no_grad():
                return float(m.slide_score(torch.as_tensor(H, device=m.w0.device))[0])

        models, z, _ = loo_normal_fits(bags, labels,
                                       lambda b, y: make_model().fit(b, y, **fit_kw),
                                       score, verbose=verbose)
        return cls(models, z, alpha)

    def member_scores(self, features):
        return torch.stack([m.slide_score(features)[0] for m in self.members])

    def forward(self, features):
        s = self.member_scores(features)
        count = (s <= self.z).sum().to(s.dtype)
        logit = self.c_max + 0.5 - count
        emb = torch.stack([m(features)[1] for m in self.members]).mean(dim=0)
        return torch.stack([torch.zeros_like(logit), logit]), F.normalize(emb, dim=-1)

    @torch.no_grad()
    def evaluate(self, features):
        features = torch.as_tensor(features, device=self.z.device)
        s = self.member_scores(features)
        c = int((s <= self.z).sum())
        return {"count": c, "p_value": (1 + c) / (len(self.z) + 1),
                "margin": float((s - self.z).mean()), "pred": int(c <= self.c_max)}

    @torch.no_grad()
    def summary(self):
        ms = [m.summary() for m in self.members]
        return {"n_normals": len(self.z), "alpha": self.alpha, "c_max": self.c_max,
                "nominal_fpr": (self.c_max + 1) / (len(self.z) + 1),
                "z_mean": float(self.z.mean()), "z_max": float(self.z.max()),
                **{f"{k}_mean": float(np.mean([s[k] for s in ms]))
                   for k in ("delta_norm", "cos_to_text", "bias", "final_loss")}}


# =====================================================================================
# Self-test
# =====================================================================================

def self_test():
    print("self_test: utils.text_topk_mil_torch")
    from utils.text_topk_mil import TextTopKMIL as NumpyTextTopK, self_test as _unused  # noqa
    rng = np.random.default_rng(0)
    d = 32

    # --- 1. autograd gradient == finite differences (float64) ------------------------
    bags = [torch.as_tensor(rng.standard_normal((50, d))) for _ in range(6)]
    y = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.float64)
    m = TextTopKMIL(np.column_stack([rng.standard_normal(d), rng.standard_normal(d)]),
                    k=3, lam=0.3, scale=20.0, dtype=torch.float64)

    with torch.no_grad():
        m.delta.copy_(0.05 * torch.randn(d, dtype=torch.float64))
        m.bias.fill_(0.2)

    def loss_now():
        S = torch.stack([m.slide_score(H)[0] for H in bags])
        return slide_bce_loss(S, y) + m.prior_penalty()

    m.zero_grad()
    loss_now().backward()
    g_auto = torch.cat([m.delta.grad, m.bias.grad.reshape(1)])
    g_num = torch.zeros_like(g_auto)
    with torch.no_grad():
        for i, (param, idx) in enumerate([(m.delta, j) for j in range(d)] + [(m.bias, None)]):
            for sgn in (1.0, -1.0):
                if idx is None:
                    param += sgn * 1e-6
                else:
                    param[idx] += sgn * 1e-6
                g_num[i] += sgn * loss_now() / 2e-6
                if idx is None:
                    param -= sgn * 1e-6
                else:
                    param[idx] -= sgn * 1e-6
    err = float((g_auto - g_num).abs().max())
    assert err < 1e-6, err
    print(f"  autograd gradient matches finite differences (max |diff| {err:.1e})")

    # --- 2. same training as the numpy reference ---------------------------------------
    w_text = np.column_stack([rng.standard_normal(d), rng.standard_normal(d)])
    tr = [rng.standard_normal((300, d)).astype(np.float32) for _ in range(20)]
    for H in tr[10:]:
        H[:3] += 2.0 * (w_text[:, 1] / np.linalg.norm(w_text[:, 1]))
    ty = np.array([0] * 10 + [1] * 10)
    ref = NumpyTextTopK(w_text, k=2, lam=0.1, scale=20.0).fit(tr, ty, epochs=60, lr=5e-3)
    tm = TextTopKMIL(w_text, k=2, lam=0.1, scale=20.0, dtype=torch.float64).fit(
        tr, ty, epochs=60, lr=5e-3)
    dD = float(np.abs(tm.delta.detach().numpy() - ref.D).max())
    db = abs(float(tm.bias.detach()) - ref.b)
    te = [rng.standard_normal((300, d)).astype(np.float32) for _ in range(30)]
    s_ref = np.array([ref.slide_score(H) for H in te])
    with torch.no_grad():
        s_tm = np.array([float(tm.slide_score(torch.as_tensor(H))[0]) for H in te])
    assert dD < 1e-6 and db < 1e-6, (dD, db)
    assert np.abs(s_ref - s_tm).max() < 1e-5
    print(f"  matches the numpy reference after 60 Adam steps "
          f"(max |dDelta| {dD:.1e}, |db| {db:.1e}, max |dS| {np.abs(s_ref - s_tm).max():.1e})")

    # --- 3. repo adapter convention ----------------------------------------------------
    H = torch.as_tensor(te[0])
    with torch.no_grad():
        logits, emb = tm(H)
        S = tm.slide_score(H)[0]
    assert logits.shape == (2,) and emb.shape == (d,)
    assert int(logits.softmax(dim=0).argmax()) == int(S > 0)
    assert abs(float(logits.softmax(dim=0)[1]) - float(torch.sigmoid(S))) < 1e-9
    print("  forward() -> (logits[0, S], embedding): argmax == (S > 0), softmax == sigmoid(S)")

    # --- 4. hooks ------------------------------------------------------------------------
    S = torch.tensor([0.2, 1.5, -0.3, 0.9, 0.1, 2.0], requires_grad=True)
    yl = torch.tensor([0, 0, 0, 1, 1, 1])
    tail_quantile_loss(S, yl, q=1.0, margin=0.5).backward()
    g = S.grad.numpy()
    assert g[1] > 0 and g[0] == 0 and g[2] == 0 and (g[3:] < 0).all()
    print("  tail_quantile_loss: gradient reaches the normal slide at the quantile "
          "(and only it) and every tumor slide")

    mim = rng.standard_normal(d)
    mm = TextTopKMIL(w_text, k=1, scale=20.0, mimic_embeddings=mim[:, None], gamma=0.5,
                     dtype=torch.float64)
    Hm = torch.as_tensor(np.vstack([mim, rng.standard_normal((5, d))]))
    base = TextTopKMIL(w_text, k=1, scale=20.0, dtype=torch.float64)
    with torch.no_grad():
        drop = base.patch_scores(Hm) - mm.patch_scores(Hm)
    assert int(drop.argmax()) == 0 and abs(float(drop[0]) - 0.5 * 20.0) < 1e-9
    lg = TextTopKMIL(w_text, k=1, scale=20.0, mimic_embeddings=mim[:, None], gamma=0.1,
                     learn_gamma=True, dtype=torch.float64)
    lg.fit(tr, ty, epochs=5, lr=5e-3)
    assert "gamma" in lg.summary()
    print("  mimic term: a patch equal to the mimic prompt loses gamma * alpha; learnable gamma trains")

    # --- 5. conformal threshold (step 4) --------------------------------------------------
    from utils.conformal import ConformalDetector
    fit_kw = dict(epochs=40, lr=5e-3)
    cm = ConformalTextTopK.fit_loo(
        tr, ty, lambda: TextTopKMIL(w_text, k=2, lam=0.1, scale=20.0, dtype=torch.float64),
        alpha=0.2, **fit_kw)
    ref_det = ConformalDetector.fit(
        tr, ty, lambda b, yy: NumpyTextTopK(w_text, k=2, lam=0.1, scale=20.0).fit(b, yy, **fit_kw),
        lambda m, H: m.slide_score(H), alpha=0.2)
    assert len(cm.members) == 10 and cm.c_max == 1           # n=10, alpha .2 -> all but one
    assert np.abs(cm.z.numpy() - ref_det.z).max() < 1e-5
    te_t = [H.copy() for H in te[:15]]
    for H in te_t[::2]:                                       # clear tumors -> some Tumor calls
        H[:3] += 5.0 * (w_text[:, 1] / np.linalg.norm(w_text[:, 1]))
    agree, preds = 0, set()
    for H in te_t:
        e = cm.evaluate(H)
        preds.add(e["pred"])
        with torch.no_grad():
            logits, emb = cm(torch.as_tensor(H))
        assert int(logits.argmax()) == e["pred"] == int(e["p_value"] <= cm.alpha)
        assert abs(float(logits[1])) >= 0.5 and emb.shape == (d,)
        r = ref_det.evaluate(H)
        agree += (r["count"] == e["count"])
        assert abs(r["margin"] - e["margin"]) < 1e-4
    assert agree == len(te_t) and preds == {0, 1}
    try:
        ConformalTextTopK.fit_loo(tr, ty, lambda: None, alpha=0.05)
        raise AssertionError("alpha < 1/(n+1) accepted")
    except ValueError:
        pass
    print(f"  ConformalTextTopK: {len(cm.members)} LOO members match the numpy detector "
          f"(counts, margins); forward argmax == (p <= alpha); alpha < 1/(n+1) refused")
    # --- 6. normal-tail ranking loss -----------------------------------------------------
    rb = [torch.as_tensor(rng.standard_normal((40, 16))) for _ in range(6)]
    ry = torch.tensor([0, 1, 0, 1, 1, 0])
    rm = TextTopKMIL(np.column_stack([rng.standard_normal(16), rng.standard_normal(16)]), k=2,
                     scale=20.0, dtype=torch.float64)
    with torch.no_grad():
        rm.delta.copy_(0.1 * torch.as_tensor(rng.standard_normal(16)))
        rm.bias.fill_(0.3)

    def rank_now(bags_=rb):
        ps_ = [rm.patch_scores(H) for H in bags_]
        return normal_tail_rank_loss(torch.stack([rm.pool(x)[0] for x in ps_]), ry, ps_, m=5, margin=1.0)

    rm.zero_grad()
    rank_now().backward()
    g_auto = rm.delta.grad.clone()
    g_num = torch.zeros(16, dtype=torch.float64)
    with torch.no_grad():
        for i in range(16):
            rm.delta[i] += 1e-6; lp = rank_now(); rm.delta[i] -= 2e-6; lm = rank_now(); rm.delta[i] += 1e-6
            g_num[i] = (lp - lm) / 2e-6
    err = float((g_auto - g_num).abs().max())
    assert err < 1e-6, err
    assert abs(float(rm.bias.grad)) < 1e-12                   # b cancels in s_neg - S_tumor
    with torch.no_grad():
        base = float(rank_now())
        rm.bias += 3.0
        assert abs(float(rank_now()) - base) < 1e-12
        rm.bias -= 3.0
        s0 = rm.patch_scores(rb[0])                          # normal slide 0
        low = int(s0.argmin())                               # outside its top-5
        hi = int(s0.argmax())                                # inside its top-5
        t_low = int(rm.patch_scores(rb[1]).argmin())         # tumor slide, not its top-2
        for b_i, p_i, should_change in ((0, low, False), (1, t_low, False), (0, hi, True)):
            mod = [H.clone() for H in rb]
            mod[b_i][p_i] -= 0.5 * (rm.w0 + rm.delta) / (rm.w0 + rm.delta).norm()
            assert (abs(float(rank_now(mod)) - base) > 1e-9) == should_change, (b_i, should_change)
    trn = [H.copy() for H in tr]
    def train_rank(model):
        with torch.no_grad():
            ps_ = [model.patch_scores(torch.as_tensor(H, dtype=torch.float64)) for H in trn]
            return float(normal_tail_rank_loss(torch.stack([model.pool(x)[0] for x in ps_]),
                                               torch.as_tensor(ty), ps_, m=16, margin=10.0))
    r0 = train_rank(TextTopKMIL(w_text, k=2, lam=0.1, scale=20.0, dtype=torch.float64).fit(trn, ty, epochs=60, lr=5e-3))
    r1 = train_rank(TextTopKMIL(w_text, k=2, lam=0.1, scale=20.0, dtype=torch.float64).fit(
        trn, ty, epochs=60, lr=5e-3, neg_weight=1.0, neg_m=16, neg_margin=10.0))
    assert r1 < r0, (r0, r1)
    print(f"  normal_tail_rank_loss: gradient matches finite differences ({err:.1e}); bias-invariant; "
          f"only top-M patches of NORMAL slides enter; training with it lowers it ({r0:.3f} -> {r1:.3f})")

    # --- 7. multi-concept model ---------------------------------------------------------
    import os
    import tempfile
    # (a) one concept per side == Text-TopK up to the constant ||t_T - t_N||
    P2 = np.column_stack([rng.standard_normal(16), rng.standard_normal(16)])
    P2n = P2 / np.linalg.norm(P2, axis=0)
    two = TextTopKMIL(P2, k=2, dtype=torch.float64)
    mc1 = MultiConceptTextTopKMIL(P2n[:, ::-1].copy(), [True, False], k=2, dtype=torch.float64)
    Hq = torch.as_tensor(rng.standard_normal((50, 16)))
    c = float(np.linalg.norm(P2n[:, 1] - P2n[:, 0]))
    assert torch.allclose(mc1.zero_shot_scores(Hq), c * two.zero_shot_scores(Hq), atol=1e-9)
    with torch.no_grad():
        assert torch.allclose(mc1.patch_scores(Hq), c * two.patch_scores(Hq), atol=1e-9)
    # (b) autograd == finite differences through the smooth maxima, with group residuals
    Pm = rng.standard_normal((16, 5))
    mc = MultiConceptTextTopKMIL(Pm, [True, True, False, False, False], k=2, lam=0.2, lam_leaf=0.7,
                                 tau=0.8, scale=20.0, dtype=torch.float64)
    with torch.no_grad():
        mc.delta.copy_(0.05 * torch.as_tensor(rng.standard_normal((16, 5))))
        mc.delta_side.copy_(0.05 * torch.as_tensor(rng.standard_normal((16, 2))))
        mc.bias.fill_(0.1)

    def mc_loss():
        ps_ = [mc.patch_scores(H) for H in rb]
        S_ = torch.stack([mc.pool(x)[0] for x in ps_])
        return (slide_bce_loss(S_, ry) + mc.prior_penalty()
                + normal_tail_rank_loss(S_, ry, ps_, m=5, margin=1.0))

    mc.zero_grad()
    mc_loss().backward()
    errs = []
    with torch.no_grad():
        for prm in (mc.delta, mc.delta_side):
            for _ in range(8):
                i, j = int(rng.integers(prm.shape[0])), int(rng.integers(prm.shape[1]))
                prm[i, j] += 1e-6; lp = mc_loss(); prm[i, j] -= 2e-6; lm = mc_loss(); prm[i, j] += 1e-6
                errs.append(abs(float(prm.grad[i, j]) - float((lp - lm) / 2e-6)))
    assert max(errs) < 1e-6, max(errs)
    # (c) the mechanism: text that names the look-alikes separates them, even zero-shot
    d_ = 32
    E = np.eye(d_)
    unit = lambda v: v / np.linalg.norm(v)
    a_T = unit(E[0] + 0.5 * E[1])
    mims = [unit(E[0] + 0.45 * E[1] + 0.6 * E[3 + j]) for j in range(3)]
    bulk = unit(E[2])

    def mk_slide(tumor, n=800, sig=0.18):
        H = bulk + sig * rng.standard_normal((n, d_))
        for j in rng.choice(3, 2, replace=False):
            ii = rng.choice(n, 8, replace=False)
            H[ii] = mims[j] + sig * rng.standard_normal((8, d_))
        if tumor:
            ii = rng.choice(n, 2, replace=False)
            H[ii] = a_T + sig * rng.standard_normal((2, d_))
        return (H / np.linalg.norm(H, axis=1, keepdims=True)).astype(np.float32)

    Nn, Tt = [mk_slide(False) for _ in range(40)], [mk_slide(True) for _ in range(40)]
    m_two = TextTopKMIL(np.column_stack([bulk, a_T]), k=1)
    m_mc = MultiConceptTextTopKMIL(np.column_stack([a_T, bulk] + mims), [True, False, False, False, False],
                                   k=1, names=["tumor", "bulk", "m0", "m1", "m2"])

    def zs_auc(m):
        f = lambda H: float(m.pool(m.zero_shot_scores(torch.as_tensor(H)))[0])
        sn = np.sort([f(H) for H in Nn])
        st = np.array([f(H) for H in Tt])
        return float(np.mean((np.searchsorted(sn, st, "left") + np.searchsorted(sn, st, "right")) / 2 / len(sn)))

    a2, am = zs_auc(m_two), zs_auc(m_mc)
    assert am > a2 + 0.15, (a2, am)
    # (d) explain(): a look-alike patch is claimed by its own normal concept
    _, nwin = m_mc.explain(torch.as_tensor(np.stack(mims).astype(np.float32)), zero_shot=True)
    assert nwin.tolist() == [2, 3, 4]
    # (e) training, conformal wrapper and the concept-file round trip
    trb = [mk_slide(False) for _ in range(6)] + [mk_slide(True) for _ in range(6)]
    tyb = np.array([0] * 6 + [1] * 6)
    fm = MultiConceptTextTopKMIL(np.column_stack([a_T, bulk] + mims), [True, False, False, False, False],
                                 k=1, lam_leaf=1.0).fit(trb, tyb, epochs=30, lr=5e-3, neg_weight=1.0,
                                                        neg_m=8, neg_margin=10.0)
    assert fm.history[-1] < fm.history[0] and "delta_norm[c2]" in fm.summary()
    cmc = ConformalTextTopK.fit_loo(
        trb, tyb, lambda: MultiConceptTextTopKMIL(np.column_stack([a_T, bulk] + mims),
                                                  [True, False, False, False, False], k=1),
        alpha=1 / 7 + 1e-9, epochs=10, lr=5e-3)
    assert len(cmc.members) == 6 and cmc.evaluate(trb[0])["pred"] in (0, 1)
    fpath = os.path.join(tempfile.mkdtemp(), "concepts.npz")
    np.savez(fpath, protos=np.column_stack([a_T, bulk] + mims).astype(np.float32),
             names=np.array(["tumor", "bulk", "m0", "m1", "m2"]), is_tumor=np.array([1, 0, 0, 0, 0], bool))
    mf = make_text_topk(concepts=fpath, k=1)
    assert isinstance(mf, MultiConceptTextTopKMIL) and mf.names[2] == "m0"
    assert isinstance(make_text_topk(np.column_stack([bulk, a_T])), TextTopKMIL)
    print(f"  multi-concept: 1+1 concepts == Text-TopK x ||t_T - t_N||; autograd == finite differences "
          f"({max(errs):.1e}); zero-shot low-WR AUC two-prototype {a2:.3f} -> multi-concept {am:.3f}; "
          f"explain/conformal/concept file OK")
    print("self_test: OK")


if __name__ == "__main__":
    self_test()
