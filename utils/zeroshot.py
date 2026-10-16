"""
utils/zeroshot.py

CONCH zero-shot pieces shared by the visualization scripts. torch and CONCH are
imported lazily, so importing this module is cheap.
"""

import numpy as np

CLASSES = ["Normal", "Tumor"]          # column order of local_data/prompts/CONCH/*.npy


def load_conch(checkpoint_path, device):
    from conch.open_clip_custom import create_model_from_pretrained
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
    return model.to(device).eval()


def load_prototypes(path, device):
    """Text prototypes as a [D, C] tensor (columns in CLASSES order)."""
    import torch
    proto = np.load(path)
    if proto.shape[0] < proto.shape[1]:           # stored [C, D]
        proto = proto.T
    if proto.shape[1] != len(CLASSES):
        raise ValueError(f"{path}: {proto.shape[1]} prototypes, expected {len(CLASSES)}")
    return torch.tensor(proto, dtype=torch.float32, device=device)


def patch_cosine(features, conch_model, prototypes, device, batch_size=4096):
    """Raw cosine similarity of every patch to every prototype, [N, C] numpy.

    Raw TRIDENT CONCH features -> visual.forward_project -> L2 normalise -> dot with the
    already-normalised prototypes (same pipeline as the analysis scripts).
    """
    import torch
    feats = torch.as_tensor(np.asarray(features), dtype=torch.float32, device=device)
    out = []
    with torch.no_grad():
        for i in range(0, feats.shape[0], batch_size):
            z = conch_model.visual.forward_project(feats[i:i + batch_size])
            out.append(torch.nn.functional.normalize(z, dim=-1) @ prototypes)
    return torch.cat(out, 0).cpu().numpy()


def logit_scale(conch_model):
    return float(conch_model.logit_scale.exp().item())


def softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def topj_pooling(sim, topj):
    """MI-Zero pooling (port of conch.downstream.zeroshot_path.topj_pooling): for each
    class independently, mean of that class's top-j patch similarities.

    Returns {j: pooled [C]}, {j: predicted class index}.
    """
    sim = np.asarray(sim, float)
    srt = -np.sort(-sim, axis=0)
    pooled = {j: srt[:min(j, len(srt))].mean(axis=0) for j in topj}
    return pooled, {j: int(np.argmax(v)) for j, v in pooled.items()}


# =====================================================================================
# Per-slide patch scorers: return the MARGIN cos(p, t_Tumor) - cos(p, t_Normal)
# (rank-preserving, no saturation); P(Tumor) = sigmoid(logit_scale * margin), which is
# exactly the 2-class softmax.
# =====================================================================================

CONCH_LOGIT_SCALE = 56.347694396972656   # exp(logit_scale) of CONCH, as in utils/adapters.py


def p_tumor_from_margin(margin, scale):
    return 1.0 / (1.0 + np.exp(-scale * np.asarray(margin, float)))


class ConchMarginScorer:
    """Scores a PatchGrid loaded with_features=True through the CONCH projection head."""

    needs_features = True

    def __init__(self, checkpoint_path, text_prototypes, batch_size=4096):
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_conch(checkpoint_path, self.device)
        self.protos = load_prototypes(text_prototypes, self.device)
        self.scale = logit_scale(self.model)
        self.batch_size = batch_size

    def __call__(self, wsi, grid):
        sim = patch_cosine(grid.features, self.model, self.protos, self.device, self.batch_size)
        return sim[:, CLASSES.index("Tumor")] - sim[:, CLASSES.index("Normal")]


class TableMarginScorer:
    """Margins from a saved per-patch table (WSI, x, y, score), matched by coordinates.
    Refuses slides where more than 1% of the h5 patches have no score (--probe tables)."""

    needs_features = False

    def __init__(self, path, scale=CONCH_LOGIT_SCALE):
        import pandas as pd
        df = pd.read_csv(path, usecols=["WSI", "x", "y", "score"])
        self.by_slide = {w: g for w, g in df.groupby("WSI")}
        self.scale = scale

    def __call__(self, wsi, grid):
        g = self.by_slide.get(wsi)
        if g is None:
            return None
        lut = {(float(x), float(y)): s for x, y, s in zip(g.x, g.y, g.score)}
        m = np.array([lut.get((float(x), float(y)), np.nan) for x, y in grid.coords])
        if np.isnan(m).mean() > 0.01:
            raise SystemExit(f"{wsi}: {int(np.isnan(m).sum())}/{len(m)} patches have no score in "
                             "the table (subsampled --probe table?). Use a full table or CONCH.")
        return np.nan_to_num(m, nan=-1.0)


def make_scorer(score_table=None, checkpoint_path=None, text_prototypes=None,
                logit_scale_value=CONCH_LOGIT_SCALE, batch_size=4096):
    if score_table:
        return TableMarginScorer(score_table, logit_scale_value)
    return ConchMarginScorer(checkpoint_path, text_prototypes, batch_size)
