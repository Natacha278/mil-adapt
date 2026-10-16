"""
utils/visualization.py

Shared drawing code for the slide visualizations (visu_ABMIL.py, visu_zs.py,
visu_zs_cossim.py). numpy + matplotlib only at import time; OpenSlide is imported only
when a thumbnail is actually read, so this module can be tested without slides.

WHAT IT FIXES COMPARED WITH THE PREVIOUS SCRIPTS
------------------------------------------------
* Patches are drawn as an image on the patch grid (draw_patch_map): each patch is a
  square of its TRUE level-0 size, at its TRUE position. The old scatter put a fixed
  `s=6` point marker at each patch's top-left corner, i.e. half a patch up-left of the
  tissue it scores, with a size unrelated to the patch (gaps on small slides).
* The patch size comes from utils.witness_rate.read_patch_size_level0 (coords attrs ->
  file attrs -> grid pitch), so the 512/1024 mix-up cannot reappear here.
* Annotation "_2" groups (non-tumor holes inside a tumor region) are drawn as dashed
  outlines, not as extra tumor.
* Every figure carries the slide's witness rate from utils.witness_rate (tumor area %,
  centre-rule %, lesion count and largest lesion with its stage).
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
import matplotlib.patheffects as pe                               # noqa: E402
from matplotlib.collections import PatchCollection                # noqa: E402
from matplotlib.patches import Polygon                            # noqa: E402

from utils.witness_rate import (                                  # noqa: E402
    Annotations, parse_camelyon16_xml, compute_slide, DEFAULT_MPP_LEVEL0)

INK, INK2 = "#0b0b0b", "#52514e"
TUMOR_FILL, TUMOR_EDGE = "#e34948", "#9b1c1c"
OUTLINE = "#ffffff"
SLIDE_EXTS = (".tif", ".svs", ".ndpi", ".mrxs", ".tiff")


# =====================================================================================
# Loading
# =====================================================================================

def find_slide_path(wsi_dir, slide_id, exts=SLIDE_EXTS):
    for ext in exts:
        p = os.path.join(wsi_dir, slide_id + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No WSI file for {slide_id} in {wsi_dir} (tried {exts})")


def load_thumbnail(slide_path, thumb_max_dim=3000):
    """(RGB uint8 array, downsample_x, downsample_y). Downsamples are measured from the
    thumbnail actually returned, so rounding of its size cannot misplace patches."""
    import openslide
    s = openslide.OpenSlide(slide_path)
    w0, h0 = s.dimensions
    scale = max(w0, h0) / thumb_max_dim
    thumb = np.asarray(s.get_thumbnail((int(w0 / scale), int(h0 / scale))).convert("RGB"))
    s.close()
    return thumb, w0 / thumb.shape[1], h0 / thumb.shape[0]


def load_annotations(xml_dir, slide_id):
    """Annotations or None when the slide has no XML (normal slides)."""
    p = os.path.join(xml_dir, f"{slide_id}.xml") if xml_dir else None
    return (parse_camelyon16_xml(p), p) if p and os.path.exists(p) else (None, None)


def witness_summary(h5_path, xml_path, gt_label, mpp_level0=DEFAULT_MPP_LEVEL0):
    """Per-slide witness rate with the correct patch footprint (utils.witness_rate)."""
    stats, _patches, lesions = compute_slide(h5_path, xml_path=xml_path,
                                             mpp_level0=mpp_level0, with_lesions=True)
    stats["gt_label"] = gt_label
    if lesions is not None and len(lesions):
        big = lesions.loc[lesions["diam_max_mm"].idxmax()]
        stats["max_lesion_diam_mm"] = float(big["diam_max_mm"])
        stats["lesion_stage"] = str(big["stage"])
    return stats


def witness_text(stats):
    """One line describing how much tumor the slide contains."""
    if stats.get("gt_label") == "Tumor" and stats.get("annotation_source") == "none":
        return "Tumor slide without an annotation file: witness rate unknown"
    if stats.get("annotation_source") == "none":
        return f"Normal slide: no annotation, witness rate 0%  ({stats['n_patches']} tissue patches)"
    s = (f"Tumor area {stats['tumor_area_pct']:.3g}% of tissue  ·  centre-rule witness rate "
         f"{stats['witness_rate_centre_pct']:.3g}% ({stats['n_tumor_centre']}/"
         f"{stats['n_patches']} patches)  ·  patches touched {stats['n_touched']}")
    if "max_lesion_diam_mm" in stats:
        s += (f"  ·  {stats['n_lesions']} lesion(s), largest {stats['max_lesion_diam_mm']:.2f} mm "
              f"({stats['lesion_stage']})")
    return s


# =====================================================================================
# Drawing
# =====================================================================================

def patch_raster(coords, patch_size, values):
    """Values on the patch grid: (2-D array with NaN where there is no patch, x0, y0).

    Row r / column c covers level-0 [x0 + c*ps, x0 + (c+1)*ps) x [y0 + r*ps, ...).
    """
    coords = np.asarray(coords, float)
    ps = float(patch_size)
    x0, y0 = coords[:, 0].min(), coords[:, 1].min()
    ix = np.rint((coords[:, 0] - x0) / ps).astype(int)
    iy = np.rint((coords[:, 1] - y0) / ps).astype(int)
    arr = np.full((iy.max() + 1, ix.max() + 1), np.nan)
    arr[iy, ix] = values
    return arr, x0, y0


def _setup_panel(ax, thumb, title):
    ax.imshow(thumb, interpolation="lanczos")
    ax.set_xlim(0, thumb.shape[1])
    ax.set_ylim(thumb.shape[0], 0)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.axis("off")


def _polys(polys, dsx, dsy, **kw):
    return PatchCollection([Polygon(np.column_stack([p[:, 0] / dsx, p[:, 1] / dsy]), closed=True)
                            for p in polys], **kw)


def draw_gt_panel(ax, thumb, ann, dsx, dsy, gt_label):
    _setup_panel(ax, thumb, "Ground truth")
    if ann is None or len(ann) == 0:
        msg = ("Normal slide: no tumor annotation" if gt_label == "Normal"
               else "Tumor slide: annotation file missing")
        ax.text(0.5, 0.03, msg, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=9, color=INK2,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.8))
        return
    ax.add_collection(_polys(ann.lesions, dsx, dsy, facecolor=TUMOR_FILL, alpha=0.4,
                             edgecolor=TUMOR_EDGE, linewidth=1.0))
    if ann.exclusions:
        ax.add_collection(_polys(ann.exclusions, dsx, dsy, facecolor="none",
                                 edgecolor=TUMOR_EDGE, linewidth=0.9, linestyle="--"))


def draw_outline(ax, ann, dsx, dsy):
    """Thin annotation outline on top of a heatmap, so alignment can be judged."""
    if ann is None or len(ann) == 0:
        return
    stroke = [pe.Stroke(linewidth=2.2, foreground=INK), pe.Normal()]
    pc = _polys(ann.lesions, dsx, dsy, facecolor="none", edgecolor=OUTLINE, linewidth=0.9)
    pc.set_path_effects(stroke)
    ax.add_collection(pc)
    if ann.exclusions:
        pe_ex = _polys(ann.exclusions, dsx, dsy, facecolor="none", edgecolor=OUTLINE,
                       linewidth=0.8, linestyle="--")
        ax.add_collection(pe_ex)


def draw_patch_map(ax, thumb, coords, patch_size, values, dsx, dsy, title,
                   cmap="viridis", vmin=None, vmax=None, alpha=0.7, ann=None, norm=None):
    """Patch values as exact squares over the thumbnail. Returns the image (colorbar)."""
    _setup_panel(ax, thumb, title)
    arr, x0, y0 = patch_raster(coords, patch_size, values)
    ny, nx = arr.shape
    extent = (x0 / dsx, (x0 + nx * patch_size) / dsx, (y0 + ny * patch_size) / dsy, y0 / dsy)
    scale = {"norm": norm} if norm is not None else {"vmin": vmin, "vmax": vmax}
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, alpha=alpha, extent=extent,
                   interpolation="nearest", **scale)
    ax.set_xlim(0, thumb.shape[1])
    ax.set_ylim(thumb.shape[0], 0)
    draw_outline(ax, ann, dsx, dsy)
    return im


def make_figure(thumb, n_panels, panel_width_in=6.5, header_lines=2):
    aspect = thumb.shape[0] / thumb.shape[1]
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(panel_width_in * n_panels, panel_width_in * aspect + 0.45 * header_lines + 0.6))
    return fig, np.atleast_1d(axes)


def add_colorbar(fig, ax, im, label):
    cax = ax.inset_axes([1.015, 0.08, 0.022, 0.84])     # outside the panel: no shrinking
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(label, fontsize=8, color=INK2)
    cb.ax.tick_params(labelsize=7, colors=INK2)
    cb.outline.set_edgecolor(INK2)
    return cb


def add_header(fig, title, lines, footer=None):
    """Bold title line, then plain lines (witness rate, prediction details)."""
    # Positions are provisional: save() moves the header just above the drawn panels
    # and the footer just below them (aspect-locked panels never fill their slots).
    texts = [fig.text(0.01, 0.99, title, fontsize=12.5, fontweight="bold", color=INK,
                      va="bottom")]
    texts += [fig.text(0.01, 0.99, line, fontsize=9.5, color=INK2, va="bottom") for line in lines]
    fig._vis_header = texts
    fig._vis_footer = (fig.text(0.01, 0.0, footer, fontsize=7.5, color=INK2, va="top")
                       if footer else None)
    fig.subplots_adjust(left=0.01, right=0.93, top=0.97, bottom=0.03, wspace=0.16)


def _pack_header_footer(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    boxes = [inv.transform(ax.get_tightbbox(r)) for ax in fig.axes]
    top = max(b[1][1] for b in boxes)
    bottom = min(b[0][1] for b in boxes)
    gap = 0.012
    y = top + gap
    for t in reversed(getattr(fig, "_vis_header", [])):
        t.set_y(y)
        h = inv.transform(t.get_window_extent(r))
        y += (h[1][1] - h[0][1]) + 0.006
    f = getattr(fig, "_vis_footer", None)
    if f is not None:
        f.set_y(bottom - gap)


def save(fig, path, dpi=200):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _pack_header_footer(fig)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Saved: {path}")


def pred_phrase(gt_label, pred_label, p_tumor):
    verdict = "correct" if pred_label == gt_label else "WRONG"
    return f"pred {pred_label}, P(Tumor) = {p_tumor:.3f}  [{verdict}]"


# =====================================================================================
# Self-test: synthetic slide, no OpenSlide / CONCH / torch
# =====================================================================================

def self_test(out_dir=None):
    import tempfile
    rng = np.random.default_rng(0)
    out_dir = out_dir or tempfile.mkdtemp()

    # 40 x 25 patch grid of 1024 level-0 px on an elliptical tissue piece
    ps, nx, ny = 1024, 40, 25
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    keep = ((gx - nx / 2) / (nx / 2)) ** 2 + ((gy - ny / 2) / (ny / 2)) ** 2 < 1
    coords = np.column_stack([gx[keep], gy[keep]]).astype(float) * ps + [3000, 5000]
    w0, h0 = 3000 + nx * ps + 3000, 5000 + ny * ps + 4000
    dsx = dsy = max(w0, h0) / 1500
    th, tw = int(h0 / dsy), int(w0 / dsx)
    yy, xx = np.mgrid[0:th, 0:tw]
    tissue = (((xx * dsx - 3000 - nx * ps / 2) / (nx * ps / 2)) ** 2 +
              ((yy * dsy - 5000 - ny * ps / 2) / (ny * ps / 2)) ** 2) < 1
    thumb = np.full((th, tw, 3), 245, np.uint8)
    thumb[tissue] = [214, 160, 200]

    # one lesion exactly on patches (10..13, 8..11) of the grid, with a hole
    L = np.array([[10, 8], [14, 8], [14, 12], [10, 12]], float) * ps + [3000, 5000]
    hole = np.array([[11.5, 9.5], [12.5, 9.5], [12.5, 10.5], [11.5, 10.5]]) * ps + [3000, 5000]
    ann = Annotations([L], [hole], ["_0"])

    # values high exactly on the lesion's patches -> must line up with the outline
    cx = (coords[:, 0] - 3000) / ps
    cy = (coords[:, 1] - 5000) / ps
    vals = 0.1 * rng.random(len(coords)) + 0.85 * ((cx >= 10) & (cx < 14) & (cy >= 8) & (cy < 12))

    arr, x0, y0 = patch_raster(coords, ps, vals)
    assert arr.shape == (ny, nx) or arr.shape[0] <= ny
    assert np.isfinite(arr).sum() == len(coords)
    # the raster cell of a patch covers exactly that patch's level-0 box
    i = int(np.argmax(vals))
    r, c = int(round((coords[i, 1] - y0) / ps)), int(round((coords[i, 0] - x0) / ps))
    assert arr[r, c] == vals[i]

    fig, axes = make_figure(thumb, 2)
    draw_gt_panel(axes[0], thumb, ann, dsx, dsy, "Tumor")
    im = draw_patch_map(axes[1], thumb, coords, ps, vals, dsx, dsy, "Synthetic score",
                        vmin=0, vmax=1, ann=ann)
    add_colorbar(fig, axes[1], im, "score")
    stats = {"gt_label": "Tumor", "annotation_source": "xml", "n_patches": len(coords),
             "tumor_area_pct": 100 * 15 / len(coords), "witness_rate_centre_pct": 100 * 15 / len(coords),
             "n_tumor_centre": 15, "n_touched": 16, "n_lesions": 1,
             "max_lesion_diam_mm": 1.4, "lesion_stage": "micro"}
    add_header(fig, "synthetic_001  ·  GT Tumor  ·  " + pred_phrase("Tumor", "Tumor", 0.97),
               [witness_text(stats)], footer="self-test")
    path = os.path.join(out_dir, "visualization_self_test.png")
    save(fig, path)
    print(f"self_test: OK ({path})")
    return path


if __name__ == "__main__":
    import sys
    self_test(sys.argv[1] if len(sys.argv) > 1 else None)
