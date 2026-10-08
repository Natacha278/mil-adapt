"""
analyze_zeroshot_score_vs_lesion_size.py

Measures CONCH zero-shot PATCH-LEVEL tumor score as a function of LESION SIZE on
CAMELYON16, per individual lesion.

WHY THIS IS NOT analyze_zeroshot_patch_accuracy.py
--------------------------------------------------
That script answers "how often is the per-patch argmax right", pooled over all
in-annotation patches of a slide. It therefore cannot distinguish "the encoder+prompt
can see a 5 mm macrometastasis but not a 0.1 mm cluster" from "it sees both equally".
This script resolves the per-patch score by the size of the lesion the patch belongs
to, which is the quantity that decides whether the slide-level tumor-ratio drop is a
POOLING problem or an ENCODER problem:

  * patch-level separation roughly FLAT in lesion size  -> the encoder sees small
    lesions; the slide-level drop comes from the MIL aggregation step.
  * patch-level separation COLLAPSES with lesion size   -> the encoder/prompt cannot
    represent small lesions; no pooling operator or attention mechanism can recover
    it, and the fix has to happen at patch granularity instead.

It also reports, per lesion, the two parameters that place that lesion in the sparse
mixture detection phase diagram:

    beta_hat = log(1 / pi) / log(N)          pi = n_patches(lesion) / N(slide)
    r_hat    = d_cohen^2 / (2 * log(N))      d_cohen = (mean_lesion - mean_surround)
                                                       / sd_surround

so each real CAMELYON16 lesion becomes one point in the (beta, r) plane and can be
compared against the Donoho-Jin detection boundary. Lesions falling BELOW the boundary
are not detectable by any test whatsoever at the current patch size -- which is a
statement about the patching configuration, not about the model.

GROUND TRUTH / DELIBERATE DIFFERENCES FROM visu_ABMIL.parse_camelyon16_xml
--------------------------------------------------------------------------
visu_ABMIL.parse_camelyon16_xml returns a FLAT list of polygons and ignores the
PartOfGroup attribute. That is fine for "is this patch inside any annotation", which is
all the existing scripts need, but it is wrong for per-lesion sizing in two ways, both
of which this file fixes (and this file does NOT monkeypatch or change the original, so
existing results stay reproducible):

  1. Flat list  -> lesions cannot be told apart, so no per-lesion area or diameter.
     Here each <Annotation> element is kept as its own lesion.
  2. PartOfGroup -> in the CAMELYON16 lesion_annotations XMLs, groups "_0" and "_1" are
     tumor and group "_2" marks NON-tumor regions enclosed inside an annotated region.
     Counting "_2" as tumor inflates lesion areas and mislabels patches. Here "_2"
     polygons are subtracted from areas and from patch occupancy.

Also recorded per patch: OCCUPANCY, the fraction of the patch's area covered by the
lesion. At 20x/512px a patch side is ~256 um, so a 0.05 mm isolated-tumor-cell cluster
covers ~0.3% of one patch; occupancy makes that within-patch dilution measurable
instead of invisible, and separates it from encoder failure.

Lesions that get ZERO patches are counted and reported rather than dropped -- a lesion
smaller than the patch grid can see is itself one of the results.

DEPENDENCIES
------------
Nothing beyond what the repo already uses (numpy, pandas, matplotlib, h5py, tqdm,
torch, sklearn, scipy). In particular NO shapely: polygon areas use the shoelace
formula and patch/lesion overlap uses matplotlib.path point sampling, both implemented
below.

USAGE
-----
    python analyze_zeroshot_score_vs_lesion_size.py \
        --xml_dir  /path/to/lesion_annotations \
        --folder   /path/to/features_root \
        --out_dir  analysis_lesion_size

Add --probe to also fit a supervised patch-level linear probe on the SAME features as
a CEILING reference (see --probe help: it is a diagnostic, not a method).
Run --self_test first; it needs neither the dataset, CONCH, nor torch.

Needs a GPU allocation only when actually scoring patches (CONCH forward_project).
--self_test and --no_score run fine on a login node.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

# Lesion-diameter bins, in mm, aligned to the AJCC/CAMELYON staging definitions:
# isolated tumor cells < 0.2, micrometastasis 0.2-2.0, macrometastasis > 2.0.
DIAM_BINS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]
STAGE_EDGES = {"ITC": (0.0, 0.2), "micro": (0.2, 2.0), "macro": (2.0, np.inf)}

TUMOR_GROUPS = ("_0", "_1")
EXCLUDE_GROUPS = ("_2",)


# =====================================================================================
# Geometry (pure numpy / matplotlib -- no shapely, unit-tested by --self_test)
# =====================================================================================

def parse_camelyon16_xml_per_lesion(xml_path):
    """Per-lesion CAMELYON16 ASAP annotations.

    Unlike visu_ABMIL.parse_camelyon16_xml (flat list, group-agnostic) this keeps each
    <Annotation> separate and splits on PartOfGroup.

    Returns (lesions, exclusions): two lists of (M, 2) float arrays of level-0 pixel
    coordinates. Annotations with an unrecognised group are treated as tumor, which is
    the conservative choice (CAMELYON16 is not perfectly consistent about group names).
    """
    root = ET.parse(xml_path).getroot()
    lesions, exclusions = [], []
    for annotation in root.iter("Annotation"):
        coords_elem = annotation.find("Coordinates")
        if coords_elem is None:
            continue
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in coords_elem.iter("Coordinate")]
        if len(pts) < 3:
            continue
        poly = np.asarray(pts, dtype=np.float64)
        group = (annotation.get("PartOfGroup") or "").strip()
        if group in EXCLUDE_GROUPS:
            exclusions.append(poly)
        else:
            lesions.append(poly)
    return lesions, exclusions


def polygon_area(poly):
    """Unsigned polygon area (shoelace), in squared input units."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _convex_hull(points):
    """Monotone-chain convex hull; returns hull vertices CCW. Pure numpy."""
    p = np.unique(points, axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for pt in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return np.asarray(lower[:-1] + upper[:-1])


def max_feret_diameter(poly):
    """Greatest dimension of the polygon = max pairwise distance over its convex hull.

    This is the diameter the AJCC definitions refer to ("greatest dimension"), so it is
    the clinically correct x-axis. The equivalent-circle diameter 2*sqrt(A/pi) is also
    reported because it is what the sparsity calculation uses.
    """
    hull = _convex_hull(poly)
    if len(hull) < 2:
        return 0.0
    d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max()))


def _inside_any(polys, pts):
    """Boolean mask: point inside at least one polygon."""
    out = np.zeros(len(pts), dtype=bool)
    for poly in polys:
        out |= MplPath(poly).contains_points(pts)
    return out


def patch_lesion_occupancy(coords, patch_size_level0, lesions, exclusions, grid=8):
    """Fraction of each patch covered by each lesion.

    coords            : (N, 2) level-0 pixel coords of patch top-left corners (TRIDENT).
    patch_size_level0 : patch side in level-0 pixels.
    grid              : patches are sampled on a grid x grid lattice, so occupancy is
                        quantised to 1/grid^2. A lesion smaller than the lattice spacing
                        can fall between sample points, so lesions whose bounding box
                        fits inside a single patch are handled analytically instead
                        (exact area ratio), which is what keeps ITC-scale lesions from
                        silently vanishing.

    Returns occ (N, n_lesions) float32 in [0, 1].
    """
    n, n_les = len(coords), len(lesions)
    occ = np.zeros((n, n_les), dtype=np.float32)
    if n == 0 or n_les == 0:
        return occ

    ps = float(patch_size_level0)
    patch_area = ps * ps
    off = (np.arange(grid) + 0.5) * ps / grid
    dx, dy = np.meshgrid(off, off, indexing="xy")
    offsets = np.column_stack([dx.ravel(), dy.ravel()])              # (grid^2, 2)
    samples = (coords[:, None, :] + offsets[None, :, :]).reshape(-1, 2)

    excluded = _inside_any(exclusions, samples) if exclusions else np.zeros(len(samples), bool)

    for j, poly in enumerate(lesions):
        lo, hi = poly.min(0), poly.max(0)
        fits_in_one_patch = np.all(hi - lo <= ps)
        if fits_in_one_patch:
            # Analytic small-lesion case: find the patch containing the lesion centroid
            # and give it the exact area ratio. Avoids the lattice missing the lesion.
            c = poly.mean(0)
            host = np.where((coords[:, 0] <= c[0]) & (c[0] < coords[:, 0] + ps) &
                            (coords[:, 1] <= c[1]) & (c[1] < coords[:, 1] + ps))[0]
            if len(host):
                area = polygon_area(poly)
                for ex in exclusions:
                    if np.all(ex.max(0) <= hi) and np.all(ex.min(0) >= lo):
                        area -= polygon_area(ex)
                occ[host[0], j] = np.clip(max(area, 0.0) / patch_area, 0.0, 1.0)
            continue
        inside = MplPath(poly).contains_points(samples) & ~excluded
        occ[:, j] = inside.reshape(n, grid * grid).mean(1)

    return occ


def lesion_geometry(lesions, exclusions, mpp_level0):
    """Per-lesion area / diameters in physical units.

    Exclusion ("_2") polygons are subtracted from a lesion when they lie inside its
    bounding box.
    """
    rows = []
    for j, poly in enumerate(lesions):
        lo, hi = poly.min(0), poly.max(0)
        area_px = polygon_area(poly)
        for ex in exclusions:
            if np.all(ex.min(0) >= lo) and np.all(ex.max(0) <= hi):
                area_px -= polygon_area(ex)
        area_px = max(area_px, 0.0)
        area_mm2 = area_px * (mpp_level0 * 1e-3) ** 2
        rows.append({
            "lesion_id": j,
            "area_mm2": area_mm2,
            "diam_eq_mm": 2.0 * np.sqrt(area_mm2 / np.pi),
            "diam_max_mm": max_feret_diameter(poly) * mpp_level0 * 1e-3,
        })
    return rows


def stage_of(diam_mm):
    for name, (lo, hi) in STAGE_EDGES.items():
        if lo <= diam_mm < hi:
            return name
    return "macro"


# =====================================================================================
# Detection-theory bookkeeping
# =====================================================================================

def detection_boundary(beta):
    """Donoho-Jin optimal detection boundary rho*(beta) for the sparse regime."""
    beta = np.asarray(beta, dtype=float)
    return np.where(beta <= 0.5, 0.0,
                    np.where(beta <= 0.75, beta - 0.5, (1.0 - np.sqrt(1.0 - beta)) ** 2))


def bonferroni_boundary(beta):
    """Boundary attained by max / top-k (Bonferroni) pooling."""
    beta = np.asarray(beta, dtype=float)
    return (1.0 - np.sqrt(np.clip(1.0 - beta, 0.0, None))) ** 2


def auc_mann_whitney(pos, neg):
    """AUC via the rank-sum identity; ties get 0.5 credit. NaN if either side empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    from scipy.stats import rankdata
    allv = np.concatenate([pos, neg])
    r = rankdata(allv)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


# =====================================================================================
# Scoring (model-dependent; torch/conch imported lazily so --self_test needs neither)
# =====================================================================================

def load_scorer(checkpoint_path, text_prototypes_path, device_str=None, batch_size=4096):
    """Returns score_fn(features[N,D]) -> (margin[N], projected[N,P]).

    margin = <proj, proto_tumor> - <proj, proto_normal>, the continuous version of the
    argmax used in analyze_zeroshot_patch_accuracy.py (same prototypes, same
    forward_project + normalize). Sign(margin) reproduces that script's prediction
    exactly; the magnitude is what this analysis needs. Scale-free, so logit_scale /
    softmax are irrelevant for AUC.
    """
    import torch
    from conch.open_clip_custom import create_model_from_pretrained

    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    proto = np.load(text_prototypes_path)
    if proto.shape[0] < proto.shape[1]:          # stored as [C, D]; want [D, C]
        proto = proto.T
    proto_t = torch.tensor(proto, dtype=torch.float32, device=device)

    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
    model = model.to(device).eval()

    # classes = ["Normal", "Tumor"] in get_project_data_camelyon16() -> tumor is col 1
    normal_idx, tumor_idx = 0, 1

    @torch.no_grad()
    def score_fn(features):
        feats = torch.as_tensor(features, dtype=torch.float32, device=device)
        chunks = []
        for i in range(0, feats.shape[0], batch_size):
            p = model.visual.forward_project(feats[i:i + batch_size])
            chunks.append(torch.nn.functional.normalize(p, dim=-1))
        proj = torch.cat(chunks, 0)
        sim = proj @ proto_t
        margin = (sim[:, tumor_idx] - sim[:, normal_idx]).cpu().numpy()
        return margin, proj.cpu().numpy()

    return score_fn


# =====================================================================================
# Per-slide pass
# =====================================================================================

def process_slide(slide_id, gt_label, h5_path, xml_path, score_fn, args):
    """Returns (patch_df, lesion_rows, n_slide_patches) for one slide."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        feats = f["features"][:]
        coords = f["coords"][:].astype(np.float64)
        ps0 = int(f.attrs.get("patch_size_level0", f.attrs.get("patch_size", 512)))

    n = len(coords)
    margin, projected = (score_fn(feats) if score_fn is not None
                         else (np.full(n, np.nan), None))

    lesions, exclusions, lesion_rows = [], [], []
    if gt_label == "Tumor" and xml_path and os.path.exists(xml_path):
        lesions, exclusions = parse_camelyon16_xml_per_lesion(xml_path)
        lesion_rows = lesion_geometry(lesions, exclusions, args.mpp_level0)

    occ = patch_lesion_occupancy(coords, ps0, lesions, exclusions, grid=args.occ_grid)
    if occ.shape[1]:
        best = occ.argmax(1)
        best_occ = occ[np.arange(n), best]
        lesion_of_patch = np.where(best_occ > args.min_occupancy, best, -1)
        occupancy = np.where(best_occ > args.min_occupancy, best_occ, 0.0)
    else:
        lesion_of_patch = np.full(n, -1)
        occupancy = np.zeros(n)

    patch_df = pd.DataFrame({
        "WSI": slide_id, "gt_label": gt_label,
        "x": coords[:, 0], "y": coords[:, 1], "patch_size_level0": ps0,
        "lesion_id": lesion_of_patch, "occupancy": occupancy,
        "is_tumor_patch": lesion_of_patch >= 0, "score": margin,
    })

    # per-lesion patch counts, including the zero-patch lesions we must not drop
    for row in lesion_rows:
        m = lesion_of_patch == row["lesion_id"]
        row.update({
            "WSI": slide_id,
            "n_patches_lesion": int(m.sum()),
            "median_occupancy": float(np.median(occupancy[m])) if m.any() else 0.0,
            "stage": stage_of(row["diam_max_mm"]),
        })

    return patch_df, lesion_rows, n, projected


# =====================================================================================
# Aggregation
# =====================================================================================

def summarise_lesions(lesion_df, patch_df, score_col="score"):
    """Slide-matched per-lesion separation + (beta_hat, r_hat)."""
    out = []
    by_slide = {w: g for w, g in patch_df.groupby("WSI")}
    for _, les in lesion_df.iterrows():
        g = by_slide.get(les["WSI"])
        rec = dict(les)
        if g is None:
            out.append(rec)
            continue
        pos = g.loc[g["lesion_id"] == les["lesion_id"], score_col].to_numpy()
        neg = g.loc[~g["is_tumor_patch"], score_col].to_numpy()          # same-slide surround
        n_slide = len(g)
        pi = len(pos) / n_slide if n_slide else np.nan
        sd = np.nanstd(neg) if len(neg) > 1 else np.nan
        d_cohen = ((np.nanmean(pos) - np.nanmean(neg)) / sd
                   if len(pos) and sd and np.isfinite(sd) and sd > 0 else np.nan)
        logN = np.log(n_slide) if n_slide > 1 else np.nan
        rec.update({
            "n_slide_patches": n_slide,
            "pi": pi,
            "auc_vs_surround": auc_mann_whitney(pos, neg),
            "mean_score_lesion": float(np.nanmean(pos)) if len(pos) else np.nan,
            "mean_score_surround": float(np.nanmean(neg)) if len(neg) else np.nan,
            "sd_score_surround": float(sd) if np.isfinite(sd) else np.nan,
            "d_cohen": d_cohen,
            "beta_hat": (np.log(1.0 / pi) / logN) if pi and pi > 0 and np.isfinite(logN) else np.nan,
            "r_hat": (d_cohen ** 2 / (2.0 * logN)) if np.isfinite(d_cohen) and np.isfinite(logN) else np.nan,
        })
        rec["above_detection_boundary"] = (
            bool(rec["r_hat"] > detection_boundary(rec["beta_hat"]))
            if np.isfinite(rec.get("r_hat", np.nan)) and np.isfinite(rec.get("beta_hat", np.nan))
            else None
        )
        out.append(rec)
    return pd.DataFrame(out)


def summarise_bins(lesion_sum):
    lab = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
           for lo, hi in zip(DIAM_BINS[:-1], DIAM_BINS[1:])]
    b = pd.cut(lesion_sum["diam_max_mm"], DIAM_BINS, labels=lab, right=False)
    g = lesion_sum.groupby(b, observed=False)
    return pd.DataFrame({
        "n_lesions": g.size(),
        "n_lesions_zero_patches": g["n_patches_lesion"].apply(lambda s: int((s == 0).sum())),
        "median_n_patches": g["n_patches_lesion"].median(),
        "median_occupancy": g["median_occupancy"].median(),
        "median_auc_vs_surround": g["auc_vs_surround"].median(),
        "q25_auc": g["auc_vs_surround"].quantile(0.25),
        "q75_auc": g["auc_vs_surround"].quantile(0.75),
        "median_d_cohen": g["d_cohen"].median(),
        "median_beta_hat": g["beta_hat"].median(),
        "median_r_hat": g["r_hat"].median(),
        "frac_above_boundary": g["above_detection_boundary"].apply(
            lambda s: np.nan if s.dropna().empty else float(s.dropna().mean())),
    }).reset_index(names="diam_max_mm_bin")


# =====================================================================================
# Figure
# =====================================================================================

def make_figure(lesion_sum, bin_sum, out_png, score_label="CONCH zero-shot margin"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axA, axB, axC = axes
    ok = lesion_sum.dropna(subset=["diam_max_mm"])
    cmap = {"ITC": "#c0392b", "micro": "#d9822b", "macro": "#1f5fa9"}
    colors = ok["stage"].map(cmap).fillna("0.5")

    # (a) separation vs lesion size -- the headline
    axA.axhline(0.5, color="0.6", lw=1, ls="--")
    axA.scatter(ok["diam_max_mm"], ok["auc_vs_surround"], s=18, c=colors, alpha=.75,
                edgecolor="none")
    # Bin medians are plotted at the MEDIAN DIAMETER OF THE LESIONS IN THE BIN, not at
    # the geometric bin centre -- an empty sub-range would otherwise put a marker where
    # no lesion exists (e.g. at 0.007 mm for the 0-0.05 bin).
    lab = [f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"
           for lo, hi in zip(DIAM_BINS[:-1], DIAM_BINS[1:])]
    binned = pd.cut(ok["diam_max_mm"], DIAM_BINS, labels=lab, right=False)
    med = (ok.groupby(binned, observed=True)
             .agg(x=("diam_max_mm", "median"), y=("auc_vs_surround", "median"))
             .dropna())
    if len(med):
        axA.plot(med["x"], med["y"], "o-", color="k", ms=5, lw=1.5, label="bin median")
    axA.set_xscale("log")
    axA.set_xlabel("lesion greatest dimension (mm)")
    axA.set_ylabel("patch AUC, lesion vs same-slide surround")
    axA.set_title("Does patch-level separation survive\nsmall lesions?", loc="left", fontsize=10)
    axA.set_ylim(0, 1.02)
    for name, col in cmap.items():
        axA.scatter([], [], c=col, s=18, label=name)
    axA.legend(frameon=False, fontsize=8, loc="lower right", ncol=2)

    # (b) within-patch dilution
    axB.scatter(ok["diam_max_mm"], ok["median_occupancy"].clip(lower=1e-4), s=18,
                c=colors, alpha=.75, edgecolor="none")
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("lesion greatest dimension (mm)")
    axB.set_ylabel("median patch occupancy")
    axB.set_title("How much of a patch the lesion\nactually fills", loc="left", fontsize=10)

    # (c) the phase diagram, with real lesions on it
    bb = np.linspace(0, 0.999, 400)
    axC.fill_between(bb, 0, detection_boundary(bb), color="#c3c9d1", alpha=.6, lw=0)
    axC.plot(bb, detection_boundary(bb), color="#1f5fa9", lw=2, label="detection boundary")
    axC.plot(bb, bonferroni_boundary(bb), color="#d9822b", lw=1.5, label="max / top-$k$")
    axC.axvline(0.5, color="0.6", lw=1, ls="--")
    fin = ok.dropna(subset=["beta_hat", "r_hat"])
    axC.scatter(fin["beta_hat"], fin["r_hat"], s=18, c=fin["stage"].map(cmap).fillna("0.5"),
                alpha=.8, edgecolor="none", zorder=5)
    # Limits must contain EVERY lesion: clipping points out of a scatter would hide
    # exactly the cases this panel exists to show. beta_hat = 1 (single-patch lesions)
    # is a real and common value, so leave headroom past it rather than on the spine.
    axC.set_xlim(0, 1.05 * max(1.0, float(fin["beta_hat"].max()) if len(fin) else 1.0))
    axC.set_ylim(0, 1.05 * max(1.0, float(fin["r_hat"].max()) if len(fin) else 1.0))
    axC.set_xlabel(r"sparsity exponent $\hat\beta$")
    axC.set_ylabel(r"signal strength $\hat r$")
    axC.set_title("Where CAMELYON16 lesions sit in the\ndetection phase diagram", loc="left",
                  fontsize=10)
    axC.legend(frameon=False, fontsize=8, loc="upper left")

    fig.text(0.005, 0.015, f"score: {score_label}; shaded region in (c) = undetectable by any test",
             fontsize=7.5, color="0.35")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================================================
# Optional supervised probe (CEILING diagnostic only)
# =====================================================================================

def add_probe_scores(patch_df, projected_by_slide, seed=0):
    """Out-of-fold supervised linear probe on the SAME projected features.

    This is NOT a method -- it uses patch-level labels that the MIL setting does not
    have. It is here to separate two causes of a low zero-shot AUC:
        probe AUC also low   -> the FEATURES do not encode small lesions (encoder limit)
        probe AUC stays high -> the features do, the TEXT PROMPT fails to read them
    Grouped by slide so no slide appears in both train and test.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    slides = [w for w in patch_df["WSI"].unique() if w in projected_by_slide]
    X = np.concatenate([projected_by_slide[w] for w in slides], 0)
    idx = np.concatenate([patch_df.index[patch_df["WSI"] == w].to_numpy() for w in slides])
    y = patch_df.loc[idx, "is_tumor_patch"].to_numpy().astype(int)
    groups = patch_df.loc[idx, "WSI"].to_numpy()

    patch_df["score_probe"] = np.nan
    n_groups = len(np.unique(groups))
    if y.sum() < 10 or (y == 0).sum() < 10 or n_groups < 2:
        print("[probe] not enough labelled patches or slides; skipped")
        return patch_df

    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, n_groups)).split(X, y, groups):
        if y[tr].sum() == 0 or (y[tr] == 0).sum() == 0:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced",
                                               random_state=seed))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.decision_function(X[te])
    patch_df.loc[idx, "score_probe"] = oof
    return patch_df


# =====================================================================================
# Self-test (no dataset, no CONCH, no torch)
# =====================================================================================

def self_test():
    print("self_test: geometry + aggregation")
    mpp = 0.25                      # um/px at level 0
    ps0 = 1024                      # 20x/512px patches -> 1024 level-0 px
    px_mm = mpp * 1e-3
    patch_mm = ps0 * px_mm
    print(f"  patch side {patch_mm*1e3:.0f} um, area {patch_mm**2:.4f} mm^2")

    # --- shoelace area on a known square -------------------------------------------
    sq = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    assert abs(polygon_area(sq) - 1e4) < 1e-6, polygon_area(sq)
    assert abs(max_feret_diameter(sq) - 100 * np.sqrt(2)) < 1e-6

    # --- a 10x10 grid of patches ----------------------------------------------------
    gx, gy = np.meshgrid(np.arange(10), np.arange(10), indexing="xy")
    coords = np.column_stack([gx.ravel(), gy.ravel()]).astype(float) * ps0

    # macro lesion: 8x8 patches (2.05 mm side, 2.9 mm greatest dimension -> macro)
    macro = np.array([[1, 1], [9, 1], [9, 9], [1, 9]], float) * ps0
    # ITC-scale lesion: 200 level-0 px across = 50 um, entirely inside patch (0,0),
    # which is outside the macro lesion
    c = np.array([0.5, 0.5]) * ps0
    th = np.linspace(0, 2 * np.pi, 33)[:-1]
    itc = c + 100.0 * np.column_stack([np.cos(th), np.sin(th)])
    # exclusion inside the macro lesion: one full patch worth at (2,2)
    excl = np.array([[2, 2], [3, 2], [3, 3], [2, 3]], float) * ps0

    lesions, exclusions = [macro, itc], [excl]
    occ = patch_lesion_occupancy(coords, ps0, lesions, exclusions, grid=16)

    # macro: 9 patches intersect, the excluded one must read ~0
    macro_occ = occ[:, 0]
    assert (macro_occ > 0.9).sum() == 63, (macro_occ > 0.9).sum()
    host_excl = np.where((coords[:, 0] == 2 * ps0) & (coords[:, 1] == 2 * ps0))[0][0]
    assert macro_occ[host_excl] < 1e-6, macro_occ[host_excl]

    # ITC: exactly one patch, with the exact small area ratio (analytic branch)
    itc_occ = occ[:, 1]
    assert (itc_occ > 0).sum() == 1, (itc_occ > 0).sum()
    expect = (np.pi * 100.0 ** 2) / (ps0 ** 2)
    assert abs(itc_occ.max() - expect) < 0.02 * expect, (itc_occ.max(), expect)
    print(f"  ITC occupancy {itc_occ.max()*100:.2f}% of one patch (expected {expect*100:.2f}%)")

    # --- geometry table -------------------------------------------------------------
    geo = lesion_geometry(lesions, exclusions, mpp)
    assert abs(geo[0]["area_mm2"] - (64 - 1) * patch_mm ** 2) < 1e-6, geo[0]
    print(f"  macro {geo[0]['diam_max_mm']:.2f} mm ({stage_of(geo[0]['diam_max_mm'])}), "
          f"ITC {geo[1]['diam_max_mm']*1e3:.0f} um ({stage_of(geo[1]['diam_max_mm'])})")
    assert stage_of(geo[0]["diam_max_mm"]) == "macro"
    assert stage_of(geo[1]["diam_max_mm"]) == "ITC"

    # --- end-to-end aggregation with a synthetic scorer ------------------------------
    rng = np.random.default_rng(0)
    best = occ.argmax(1); best_occ = occ[np.arange(len(coords)), best]
    lid = np.where(best_occ > 0, best, -1)
    score = 2.5 * best_occ + rng.normal(0, 1.0, len(coords))   # separable in occupancy
    patch_df = pd.DataFrame({"WSI": "tumor_test", "gt_label": "Tumor",
                             "x": coords[:, 0], "y": coords[:, 1],
                             "lesion_id": lid, "occupancy": best_occ,
                             "is_tumor_patch": lid >= 0, "score": score})
    for g in geo:
        m = lid == g["lesion_id"]
        g.update({"WSI": "tumor_test", "n_patches_lesion": int(m.sum()),
                  "median_occupancy": float(np.median(best_occ[m])) if m.any() else 0.0,
                  "stage": stage_of(g["diam_max_mm"])})
    ls = summarise_lesions(pd.DataFrame(geo), patch_df)
    assert set(["beta_hat", "r_hat", "auc_vs_surround"]) <= set(ls.columns)
    assert ls.loc[ls["stage"] == "macro", "auc_vs_surround"].iloc[0] > 0.8
    bs = summarise_bins(ls)
    assert bs["n_lesions"].sum() == 2, bs
    print(ls[["stage", "diam_max_mm", "n_patches_lesion", "median_occupancy",
              "auc_vs_surround", "beta_hat", "r_hat", "above_detection_boundary"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))

    # --- boundary sanity ------------------------------------------------------------
    assert detection_boundary(0.3) == 0.0
    assert abs(detection_boundary(0.6) - 0.1) < 1e-12
    assert abs(detection_boundary(0.9) - (1 - np.sqrt(0.1)) ** 2) < 1e-12
    assert bonferroni_boundary(0.6) > detection_boundary(0.6)   # max is suboptimal here
    print("self_test: OK")


# =====================================================================================
# Main
# =====================================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--xml_dir", help="Directory containing lesion_annotations/*.xml.")
    p.add_argument("--folder", help="Passed to utils.utils.load_data for the slide list + GT.")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--checkpoint_path",
                   default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--mpp_level0", type=float, default=0.243,
                   help="Microns per pixel at level 0. CAMELYON16 is scanned at 40x; "
                        "0.243 (3DHistech) and 0.226 (Hamamatsu) are the two values in "
                        "the cohort. Lesion sizes in mm scale linearly with this, so set "
                        "it per scanner if you need exact diameters.")
    p.add_argument("--occ_grid", type=int, default=8,
                   help="Patch occupancy sampling lattice (occ_grid^2 points per patch).")
    p.add_argument("--min_occupancy", type=float, default=0.0,
                   help="Occupancy above which a patch counts as belonging to a lesion. "
                        "0.0 keeps every patch a lesion touches, which is what you want "
                        "for small lesions; raise it to reproduce a stricter convention.")
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None, help="Process only the first N slides.")
    p.add_argument("--probe", action="store_true",
                   help="Also fit a supervised patch-level linear probe as a CEILING "
                        "reference (uses patch labels; diagnostic only, not a method).")
    p.add_argument("--probe_neg_per_slide", type=int, default=300,
                   help="Cap on normal patches kept per slide for the probe.")
    p.add_argument("--no_score", action="store_true",
                   help="Geometry only: skip CONCH entirely. Gives lesion sizes, patch "
                        "counts, occupancy and beta_hat (no r_hat). Runs on a login node.")
    p.add_argument("--save_patches", action="store_true",
                   help="Also write the full per-patch table (csv.gz).")
    p.add_argument("--out_dir", default="analysis_lesion_size")
    p.add_argument("--exp_name", default="")
    p.add_argument("--self_test", action="store_true",
                   help="Validate geometry + aggregation on synthetic data and exit.")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.xml_dir or not args.folder:
        p.error("--xml_dir and --folder are required (or use --self_test)")

    os.makedirs(args.out_dir, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from utils.utils import load_data
    from tqdm import tqdm

    classes = ["Normal", "Tumor"]
    _X, Y, WSI = load_data(folder=args.folder, project=args.project,
                           encoder=args.encoder, classes=classes)
    if args.limit:
        WSI, Y = WSI[:args.limit], Y[:args.limit]

    score_fn = None if args.no_score else load_scorer(
        args.checkpoint_path, args.text_prototypes, batch_size=args.batch_size)

    patch_frames, lesion_rows_all, projected_by_slide = [], [], {}
    n_missing_h5 = n_tumor_no_xml = 0
    rng = np.random.default_rng(0)

    for i in tqdm(range(len(WSI)), desc="score vs lesion size"):
        slide_id, gt_label = WSI[i], classes[Y[i]]
        h5_path = os.path.join(args.h5_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            n_missing_h5 += 1
            continue
        xml_path = os.path.join(args.xml_dir, f"{slide_id}.xml")
        if gt_label == "Tumor" and not os.path.exists(xml_path):
            n_tumor_no_xml += 1
            continue

        pdf, lrows, _n, projected = process_slide(slide_id, gt_label, h5_path,
                                                  xml_path, score_fn, args)
        patch_frames.append(pdf)
        lesion_rows_all.extend(lrows)

        if args.probe and projected is not None:
            keep = np.where(pdf["is_tumor_patch"].to_numpy())[0]
            neg = np.where(~pdf["is_tumor_patch"].to_numpy())[0]
            if len(neg) > args.probe_neg_per_slide:
                neg = rng.choice(neg, args.probe_neg_per_slide, replace=False)
            sel = np.sort(np.concatenate([keep, neg]))
            patch_frames[-1] = pdf.iloc[sel].copy()
            projected_by_slide[slide_id] = projected[sel]

    if not patch_frames:
        raise SystemExit("no slides processed -- check --h5_dir / --folder")

    patch_df = pd.concat(patch_frames, ignore_index=True)
    lesion_df = pd.DataFrame(lesion_rows_all)

    if args.probe and projected_by_slide:
        patch_df = add_probe_scores(patch_df, projected_by_slide)

    lesion_sum = summarise_lesions(lesion_df, patch_df, "score")
    if "score_probe" in patch_df.columns and patch_df["score_probe"].notna().any():
        probe_sum = summarise_lesions(lesion_df, patch_df, "score_probe")
        lesion_sum["auc_probe_vs_surround"] = probe_sum["auc_vs_surround"].to_numpy()
    bin_sum = summarise_bins(lesion_sum)

    tag = f"{args.project}_{args.encoder}_{args.exp_name}"
    les_csv = os.path.join(args.out_dir, f"{tag}_per_lesion.csv")
    bin_csv = os.path.join(args.out_dir, f"{tag}_bin_summary.csv")
    png = os.path.join(args.out_dir, f"{tag}_score_vs_lesion_size.png")
    lesion_sum.to_csv(les_csv, index=False)
    bin_sum.to_csv(bin_csv, index=False)
    if args.save_patches:
        patch_df.to_csv(os.path.join(args.out_dir, f"{tag}_per_patch.csv.gz"),
                        index=False, compression="gzip")
    if not args.no_score:
        make_figure(lesion_sum, bin_sum, png)

    tum = patch_df[patch_df["is_tumor_patch"]]["score"].to_numpy()
    nor = patch_df[~patch_df["is_tumor_patch"]]["score"].to_numpy()
    n_zero = int((lesion_sum["n_patches_lesion"] == 0).sum())

    # beta=1/2 wall for the median slide at this patch size: pi = N^-1/2
    med_n = float(lesion_sum["n_slide_patches"].median()) if len(lesion_sum) else np.nan
    print("\n" + "=" * 78)
    print(f"Zero-shot patch score vs lesion size  [{args.project}/{args.encoder}]")
    print("=" * 78)
    print(f"slides processed            : {patch_df['WSI'].nunique()}"
          f"  (missing h5: {n_missing_h5}, tumor w/o xml: {n_tumor_no_xml})")
    print(f"lesions                     : {len(lesion_sum)}"
          f"  (ITC {int((lesion_sum['stage']=='ITC').sum())}, "
          f"micro {int((lesion_sum['stage']=='micro').sum())}, "
          f"macro {int((lesion_sum['stage']=='macro').sum())})")
    print(f"lesions with ZERO patches   : {n_zero}"
          f"   <- invisible to the patch grid at this patch size")
    if np.isfinite(med_n):
        print(f"median patches per slide N  : {med_n:.0f}  ->  beta=1/2 wall at "
              f"pi = {med_n**-0.5:.4f} of tissue patches")
    if not args.no_score:
        print(f"pooled patch AUC (all tumor vs all normal patches): "
              f"{auc_mann_whitney(tum, nor):.4f}")
    print("-" * 78)
    cols = ["diam_max_mm_bin", "n_lesions", "n_lesions_zero_patches", "median_n_patches",
            "median_occupancy", "median_auc_vs_surround", "median_d_cohen",
            "median_beta_hat", "median_r_hat", "frac_above_boundary"]
    print(bin_sum[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("=" * 78)
    print(f"per-lesion -> {les_csv}\nbin summary -> {bin_csv}"
          + ("" if args.no_score else f"\nfigure      -> {png}"))
    print("\nRead it this way: median_auc_vs_surround roughly flat across bins means the\n"
          "encoder sees small lesions and the slide-level tumor-ratio drop is a POOLING\n"
          "problem. Falling toward 0.5 in the small bins means it is an ENCODER problem.\n"
          "Compare median_occupancy to tell within-patch dilution apart from either, and\n"
          "frac_above_boundary for how many lesions are detectable even in principle.")


if __name__ == "__main__":
    main()
