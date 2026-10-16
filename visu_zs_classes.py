"""
visu_zs_classes.py

CONCH zero-shot tissue-class map of a slide: every patch is assigned to the class whose
text prototype it is most similar to (max cosine similarity), drawn at its true
level-0 position and size. Tumor slides also get the annotated tumor contour, and the
class composition is reported for the whole slide, inside the tumor and outside it.

Classes come from a JSON file, either a plain dict
    {"lymphocytes": "Dense sheets of small, round ...", "germinal_center": "...", ...}
or with several phrasings per class (averaged, like create_CONCH_prompt_feat.py) and
optional colors:
    {"classes": {"lymphocytes": ["text 1", "text 2"], ...},
     "colors":  {"lymphocytes": "#2a78d6", ...}}
Optional "groups" (e.g. normal vs tumor) for more than 8 classes:
    {"classes": {...}, "groups": {"normal": ["lymphocytes", ...], "tumor": ["solid_carcinoma_nests", ...]}}
With groups, colors come from one hue family per group (normal: blues / greens / violets,
up to 7 classes; tumor: reds / yellows / magentas, up to 6), the figure adds a two-color
group map (always readable, whatever the number of classes), and the bars and CSV get
group subtotals. Without groups, classes take a fixed 12-color order (the first 8 are a
validated categorical palette; beyond 8 some pairs are hard to tell apart, so read the
labelled bars). Explicit "colors" always win.

Text prototypes: CONCH text encoder, each phrasing L2-normalised, averaged per class,
re-normalised (same as conch.downstream.zeroshot_path.zero_shot_classifier with the
template "CLASSNAME"). Patch embeddings: the MIL-format .npy (CONCH projection head,
L2-normalised, same order as the TRIDENT .h5) when --folder is given; otherwise the raw
.h5 features are projected with CONCH (visual.forward_project), as in visu_zs.py.

--min_margin m > 0 marks patches whose best and second-best similarities differ by less
than m as "ambiguous" (grey).

Outputs, per slide: <out_dir>/<slide>_zs_classes<tag>.png (class map with the tumor contour
| composition bars; --show_gt adds a separate ground-truth panel); and one CSV for all slides,
<out_dir>/zs_class_composition<tag>.csv: slide, gt, class, n / % of all patches, of
tumor patches (patch centre inside the annotation), of non-tumor patches.

    python visu_zs_classes.py --slide_ids test_001 tumor_009 normal_001 \\
        --class_json local_data/prompts/templates/camelyon_normal_classes.json \\
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \\
        --wsi_dir $SCRATCH/camelyon16/images --xml_dir $SCRATCH/camelyon16/annotations

    python visu_zs_classes.py --self_test        # synthetic slide, no CONCH / OpenSlide
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PALETTE_EXT = PALETTE + ["#184f95", "#a3261f", "#9a6a00", "#0f7a6a"]          # 9-12: weaker separation
FAMILY_PALETTES = {                                                           # within-family order
    "normal": ["#2a78d6", "#1baf7a", "#6250d6", "#008300", "#184f95", "#9085e9", "#5598e7"],
    "tumor": ["#e34948", "#eda100", "#e87ba4", "#a3261f", "#9a6a00", "#eb6834"],
}
GROUP_COLORS = {"normal": "#2a78d6", "tumor": "#e34948"}
OTHER_GROUP_COLORS = ["#eda100", "#6250d6", "#1baf7a"]
AMBIGUOUS_COLOR, AMBIGUOUS = "#9e9e9e", "ambiguous"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# =====================================================================================
# Classes and assignment
# =====================================================================================

def _family(group):
    g = group.lower()
    return "tumor" if "tumor" in g or "tumour" in g or "cancer" in g else ("normal" if "normal" in g else None)


def load_class_spec(path):
    """-> (names, texts per class (list of lists), colors, groups per class or None)."""
    with open(path) as f:
        spec = json.load(f)
    classes = spec["classes"] if isinstance(spec, dict) and "classes" in spec else spec
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"{path}: expected a non-empty {{class: text}} dict (or {{'classes': ...}})")
    names = list(classes)
    texts = [[v] if isinstance(v, str) else list(v) for v in classes.values()]
    if any(not t or not all(isinstance(s, str) and s.strip() for s in t) for t in texts):
        raise ValueError(f"{path}: every class needs at least one non-empty text")
    user = spec.get("colors", {}) if "classes" in spec else {}
    gspec = spec.get("groups") if "classes" in spec else None
    groups = None
    if gspec:
        member = {c: g for g, cs in gspec.items() for c in cs}
        unknown = sorted(set(member) - set(names))
        if unknown:
            raise ValueError(f"{path}: groups name unknown classes {unknown}")
        missing = [n for n in names if n not in member]
        if missing:
            raise ValueError(f"{path}: classes without a group {missing}")
        groups = [member[n] for n in names]
    colors = []
    if groups:
        used = {}
        for n, g in zip(names, groups):
            fam = FAMILY_PALETTES.get(_family(g))
            i = used.get(g, 0)
            used[g] = i + 1
            if n in user:
                colors.append(user[n])
            elif fam is not None and i < len(fam):
                colors.append(fam[i])
            else:
                colors.append(None)                       # filled below from the extended palette
        spare = [c for c in PALETTE_EXT if c not in colors]
        for j, c in enumerate(colors):
            if c is None:
                if not spare:
                    raise ValueError(f"{path}: too many classes for the palettes; give 'colors'")
                colors[j] = spare.pop(0)
    else:
        if len(names) > len(PALETTE_EXT) and len(user) < len(names):
            raise ValueError(f"{len(names)} classes but only {len(PALETTE_EXT)} palette colors: "
                             f"add 'groups' or 'colors' to the JSON, or merge classes")
        colors = [user.get(n, PALETTE_EXT[i]) for i, n in enumerate(names)]
        if len(names) > len(PALETTE):
            print(f"note: {len(names)} classes > {len(PALETTE)}: colors 9+ are less distinct; "
                  f"add 'groups' (normal / tumor) for family colors and a group map")
    return names, texts, colors, groups


def group_colors(group_names):
    out, k = [], 0
    for g in group_names:
        fam = _family(g)
        if fam in GROUP_COLORS:
            out.append(GROUP_COLORS[fam])
        else:
            out.append(OTHER_GROUP_COLORS[k % len(OTHER_GROUP_COLORS)])
            k += 1
    return out


def encode_classes(conch_model, texts, device):
    """CONCH text prototypes, [D, C] float32 numpy, unit columns."""
    import torch
    import torch.nn.functional as F
    from conch.open_clip_custom import get_tokenizer, tokenize
    tok = get_tokenizer()
    cols = []
    with torch.no_grad():
        for phrasings in texts:
            e = F.normalize(conch_model.encode_text(tokenize(tok, phrasings).to(device)), dim=-1)
            m = e.mean(0)
            cols.append((m / m.norm()).cpu().numpy())
    return np.stack(cols, 1).astype(np.float32)


def assign_classes(sim, min_margin=0.0):
    """sim [N, C] -> class index per patch (C = ambiguous when top-2 gap < min_margin)."""
    idx = sim.argmax(1)
    if min_margin > 0 and sim.shape[1] > 1:
        top2 = np.sort(sim, 1)[:, -2:]
        idx = np.where(top2[:, 1] - top2[:, 0] < min_margin, sim.shape[1], idx)
    return idx


def composition(idx, n_classes, inside=None):
    """Counts per class (+ ambiguous): all patches, inside tumor, outside tumor."""
    k = n_classes + 1
    out = {"all": np.bincount(idx, minlength=k)}
    if inside is not None:
        out["tumor"] = np.bincount(idx[inside], minlength=k)
        out["non_tumor"] = np.bincount(idx[~inside], minlength=k)
    return out


# =====================================================================================
# Figure
# =====================================================================================

def draw_composition(ax, names, colors, comp, class_groups=None):
    groups = [("all", "all patches", 1.0, None)]
    if "tumor" in comp and comp["tumor"].sum():
        groups += [("tumor", "inside tumor", 1.0, "////"), ("non_tumor", "outside tumor", 0.45, None)]
    k = len(names) + 1
    show = [i for i in range(k) if i < len(names) or comp["all"][i] > 0]   # ambiguous only if used
    if class_groups:                                                       # group by group
        order = list(dict.fromkeys(class_groups))
        show.sort(key=lambda i: (order.index(class_groups[i]) if i < len(names) else len(order), i))
    labels = names + [AMBIGUOUS]
    cols = colors + [AMBIGUOUS_COLOR]
    h = 0.8 / len(groups)
    y = np.arange(len(show))[::-1]
    for g, (key, lab, alpha, hatch) in enumerate(groups):
        tot = max(1, comp[key].sum())
        pct = 100 * comp[key][show] / tot
        yy = y + 0.4 - h * (g + 0.5)
        ax.barh(yy, pct, height=h * 0.92, color=[cols[i] for i in show], alpha=alpha,
                hatch=hatch, edgecolor="white" if hatch else "none", linewidth=0)
        for yi, v in zip(yy, pct):
            if v > 0:
                ax.text(v + 1, yi, f"{v:.1f}%" if v >= 0.05 else "<0.1%", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in show], fontsize=8.5, color=INK)
    if class_groups:                                                       # thin separators
        prev = None
        for yi, i in zip(y, show):
            g = class_groups[i] if i < len(names) else None
            if prev is not None and g != prev:
                ax.axhline(yi + 0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
            prev = g
    ax.set_xlim(0, 112)
    ax.set_xlabel("% of patches", fontsize=8.5, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=7.5)
    if len(groups) > 1:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="#777777", label="all patches"),
                           Patch(facecolor="#777777", hatch="////", edgecolor="white", label="inside tumor"),
                           Patch(facecolor="#777777", alpha=0.45, label="outside tumor")],
                  fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    ax.set_title("Class composition", fontsize=10, color=INK, loc="left")


def render(slide_id, gt, thumb, dsx, dsy, coords, ps, idx, names, colors, ann, stats, comp,
           out_path, alpha=0.75, footer="", show_gt=False, groups=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from utils import visualization as V

    k = len(names)
    aspect = thumb.shape[0] / thumb.shape[1]
    w = 6.0
    gnames = list(dict.fromkeys(groups)) if groups else []
    n_maps = 1 + int(show_gt) + int(bool(groups))
    label_in = 0.075 * max(len(n) for n in names + [AMBIGUOUS])       # room for the bar labels
    fig = plt.figure(figsize=(n_maps * w + 3.4 + label_in, max(w * aspect, 0.62 * (k + 1)) + 1.8))
    gs = fig.add_gridspec(1, n_maps + 1, width_ratios=[1] * n_maps + [3.4 / w],
                          wspace=0.06 + label_in / w)
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_maps + 1)]
    if show_gt:
        V.draw_gt_panel(axes[0], thumb, ann, dsx, dsy, gt)
    ax1, ax2 = axes[int(show_gt)], axes[-1]
    if groups:
        axg = axes[int(show_gt) + 1]
        gcols = group_colors(gnames)
        gidx = np.array([gnames.index(groups[i]) if i < k else len(gnames) for i in idx], float)
        V.draw_patch_map(axg, thumb, coords, ps, gidx, dsx, dsy, "Zero-shot group",
                         cmap=ListedColormap(gcols + [AMBIGUOUS_COLOR]),
                         norm=BoundaryNorm(np.arange(len(gnames) + 2) - 0.5, len(gnames) + 1),
                         alpha=alpha, ann=ann if gt == "Tumor" else None)
        gh = [Patch(facecolor=c, label=f"{g} ({100 * np.isin(idx, [i for i in range(k) if groups[i] == g]).mean():.1f}%)")
              for g, c in zip(gnames, gcols)]
        if gt == "Tumor" and ann is not None and len(ann):
            gh.append(Patch(facecolor="none", edgecolor=INK, label="tumor annotation"))
        axg.legend(handles=gh, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=min(4, len(gh)),
                   fontsize=7.5, frameon=False)
    cmap = ListedColormap(colors + [AMBIGUOUS_COLOR])
    norm = BoundaryNorm(np.arange(k + 2) - 0.5, k + 1)
    V.draw_patch_map(ax1, thumb, coords, ps, idx.astype(float), dsx, dsy,
                     "Zero-shot class (max cosine similarity)", cmap=cmap, norm=norm, alpha=alpha,
                     ann=ann if gt == "Tumor" else None)
    present = [i for i in range(k + 1) if comp["all"][i] > 0]
    if groups:
        present.sort(key=lambda i: (gnames.index(groups[i]) if i < k else len(gnames), i))
    handles = [Patch(facecolor=(colors + [AMBIGUOUS_COLOR])[i], label=(names + [AMBIGUOUS])[i])
               for i in present]
    if gt == "Tumor" and ann is not None and len(ann):
        handles.append(Patch(facecolor="none", edgecolor=INK, label="tumor annotation"))
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=min(4, len(handles)),
               fontsize=7.5, frameon=False)
    draw_composition(ax2, names, colors, comp, groups)

    lines = [V.witness_text(stats)] if stats else []
    if "tumor" in comp and comp["tumor"].sum():
        allnames = names + [AMBIGUOUS]
        t = comp["tumor"]
        top = np.argsort(-t)[:3]
        txt = "inside the tumor annotation: "
        if groups:
            txt += ", ".join(f"{g} group {100 * sum(t[i] for i in range(k) if groups[i] == g) / t.sum():.1f}%"
                             for g in gnames) + "  ·  top classes: "
        lines.append(txt + ", ".join(f"{allnames[i]} {100 * t[i] / t.sum():.1f}%" for i in top if t[i] > 0))
    V.add_header(fig, f"{slide_id}  ·  GT {gt}  ·  zero-shot class map ({k} classes)", lines,
                 footer=footer)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.08)
    V.save(fig, out_path)


# =====================================================================================
# Per slide
# =====================================================================================

def process_slide(slide_id, gt, feats_unit, coords, ps, protos, names, colors, ann, stats,
                  thumb, dsx, dsy, args, footer="", groups=None):
    from utils.witness_rate import patch_centre_in_polygons
    sim = feats_unit @ protos
    idx = assign_classes(sim, args.min_margin)
    inside = (patch_centre_in_polygons(coords, ps, ann)
              if gt == "Tumor" and ann is not None and len(ann) else None)
    comp = composition(idx, len(names), inside)
    tag = f"_{args.exp_name}" if args.exp_name else ""
    render(slide_id, gt, thumb, dsx, dsy, coords, ps, idx, names, colors, ann, stats, comp,
           os.path.join(args.out_dir, f"{slide_id}_zs_classes{tag}.png"), alpha=args.alpha,
           footer=footer, show_gt=getattr(args, "show_gt", False), groups=groups)

    def row(label, group, sel):
        r = {"slide_id": slide_id, "gt_label": gt, "class": label, "group": group,
             "n_all": int(comp["all"][sel].sum()),
             "pct_all": 100 * comp["all"][sel].sum() / max(1, comp["all"].sum())}
        if inside is not None:
            r.update({"n_tumor": int(comp["tumor"][sel].sum()),
                      "pct_tumor": 100 * comp["tumor"][sel].sum() / max(1, comp["tumor"].sum()),
                      "n_non_tumor": int(comp["non_tumor"][sel].sum()),
                      "pct_non_tumor": 100 * comp["non_tumor"][sel].sum() / max(1, comp["non_tumor"].sum())})
        return r

    rows = []
    for i, n in enumerate(names + [AMBIGUOUS]):
        if i == len(names) and comp["all"][i] == 0:
            continue
        rows.append(row(n, groups[i] if groups and i < len(names) else "", [i]))
    for g in (dict.fromkeys(groups) if groups else []):               # group subtotals
        rows.append(row(f"[{g}]", g, [i for i in range(len(names)) if groups[i] == g]))
    return rows, idx, sim


def main_run(args):
    import h5py
    import torch
    from utils import visualization as V
    from utils import zeroshot as Z
    from utils.witness_rate import read_patch_size_level0, DEFAULT_MPP_LEVEL0

    names, texts, colors, groups = load_class_spec(args.class_json)
    meta = pd.read_csv(args.csv_path or f"./local_data/csv/{args.project}.csv")
    gts = dict(zip(meta.WSI, meta.GT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conch = Z.load_conch(args.checkpoint_path, device)
    protos = encode_classes(conch, texts, device)
    print("classes: " + ", ".join(f"{n} ({len(t)} text{'s' if len(t) > 1 else ''})"
                                  for n, t in zip(names, texts)))
    cos = protos.T @ protos
    print("text-prototype cosine similarities:\n" + pd.DataFrame(cos, index=names, columns=names)
          .to_string(float_format=lambda v: f"{v:.2f}"))
    rows = []
    for sid in args.slide_ids:
        if sid not in gts:
            print(f"{sid}: not in the project CSV, skipped")
            continue
        gt = gts[sid]
        h5_path = os.path.join(args.h5_dir, f"{sid}.h5")
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:].astype(float)
            ps = read_patch_size_level0(f, coords)
            if args.folder:
                feats = np.load(os.path.join(args.folder, args.project, args.encoder, f"{sid}.npy"))
                if len(feats) != len(coords):
                    raise SystemExit(f"{sid}: {len(feats)} .npy patches vs {len(coords)} .h5 coords")
                feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            else:
                with torch.no_grad():
                    z = conch.visual.forward_project(torch.as_tensor(f["features"][:], dtype=torch.float32,
                                                                     device=device))
                    feats = torch.nn.functional.normalize(z, dim=-1).cpu().numpy()
        ann, xml_path = V.load_annotations(args.xml_dir, sid)
        stats = V.witness_summary(h5_path, xml_path, gt, args.mpp_level0 or DEFAULT_MPP_LEVEL0)
        thumb, dsx, dsy = V.load_thumbnail(V.find_slide_path(args.wsi_dir, sid), args.thumb_max_dim)
        footer = (f"classes from {os.path.basename(args.class_json)}; patch = argmax cosine similarity"
                  + (f", ambiguous if top-2 gap < {args.min_margin:g}" if args.min_margin > 0 else "")
                  + f"; patches at true level-0 size ({ps:g} px); inside tumor = patch centre in annotation")
        r, idx, _ = process_slide(sid, gt, feats.astype(np.float32), coords, ps, protos, names, colors,
                                  ann, stats, thumb, dsx, dsy, args, footer, groups)
        rows += r
        print(f"{sid} ({gt}): " + ", ".join(f"{x['class']} {x['pct_all']:.1f}%" for x in r))
    if rows:
        tag = f"_{args.exp_name}" if args.exp_name else ""
        path = os.path.join(args.out_dir, f"zs_class_composition{tag}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"-> {path}")


# =====================================================================================
# Self-test: synthetic slide with known classes, no CONCH / OpenSlide
# =====================================================================================

def self_test(out_dir):
    import tempfile
    from utils.witness_rate import Annotations
    from utils import visualization as V
    rng = np.random.default_rng(0)
    out_dir = out_dir or tempfile.mkdtemp()
    tmp = tempfile.mkdtemp()

    # class spec: plain dict form and full form must agree
    spec = {"lymphocytes": "small dark cells", "germinal_center": ["pale zone", "centroblasts"],
            "adipose_tissue": "fat", "tumor": "carcinoma nests"}
    p1, p2 = os.path.join(tmp, "a.json"), os.path.join(tmp, "b.json")
    json.dump(spec, open(p1, "w"))
    json.dump({"classes": spec, "colors": {"tumor": "#e34948"}}, open(p2, "w"))
    n1, t1, c1, g1 = load_class_spec(p1)
    n2, t2, c2, g2 = load_class_spec(p2)
    assert g1 is None and g2 is None
    assert n1 == n2 == list(spec) and t1[1] == ["pale zone", "centroblasts"] and t1[0] == ["small dark cells"]
    assert c1[:3] == PALETTE[:3] and c2[3] == "#e34948"
    json.dump({"a": ""}, open(p1, "w"))
    try:
        load_class_spec(p1)
        raise AssertionError("empty text accepted")
    except ValueError:
        pass

    # synthetic slide: 30 x 20 grid, regions of known class, one lesion
    names, colors = n2, c2
    d, ps, nx, ny = 32, 1024, 30, 20
    protos = np.linalg.qr(rng.standard_normal((d, len(names))))[0].astype(np.float32)   # orthonormal
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny))
    coords = np.column_stack([gx.ravel(), gy.ravel()]).astype(float) * ps + [2000, 3000]
    cls = np.zeros(len(coords), int)                                # lymphocytes background
    cls[(gx.ravel() < 8)] = 2                                       # fat on the left
    cls[((gx.ravel() - 18) ** 2 + (gy.ravel() - 6) ** 2) < 9] = 1   # a germinal centre
    lesion = (gx.ravel() >= 20) & (gx.ravel() < 24) & (gy.ravel() >= 12) & (gy.ravel() < 16)
    cls[lesion] = 3
    feats = protos.T[cls] + 0.15 * rng.standard_normal((len(coords), d))
    feats = (feats / np.linalg.norm(feats, axis=1, keepdims=True)).astype(np.float32)
    L = np.array([[20, 12], [24, 12], [24, 16], [20, 16]], float) * ps + [2000, 3000]
    ann = Annotations([L], [], ["_0"])
    w0, h0 = 2000 + nx * ps + 2000, 3000 + ny * ps + 2000
    dsx = dsy = max(w0, h0) / 1200
    th, tw = int(h0 / dsy), int(w0 / dsx)
    thumb = np.full((th, tw, 3), 243, np.uint8)                     # glass
    yy, xx = np.mgrid[0:th, 0:tw]
    x_l0, y_l0 = xx * dsx, yy * dsy
    tissue = ((x_l0 >= 2000) & (x_l0 < 2000 + nx * ps) & (y_l0 >= 3000) & (y_l0 < 3000 + ny * ps))
    shade = rng.normal(0, 12, (th, tw, 1))
    pink = np.clip(np.array([200, 120, 175]) + shade, 0, 255)       # H&E-like texture
    thumb[tissue] = pink[tissue].astype(np.uint8)
    stats = {"gt_label": "Tumor", "annotation_source": "xml", "n_patches": len(coords),
             "tumor_area_pct": 100 * lesion.mean(), "witness_rate_centre_pct": 100 * lesion.mean(),
             "n_tumor_centre": int(lesion.sum()), "n_touched": int(lesion.sum()), "n_lesions": 1,
             "max_lesion_diam_mm": 1.4, "lesion_stage": "micro"}
    args = argparse.Namespace(min_margin=0.0, exp_name="", out_dir=out_dir, alpha=0.75)
    rows, idx, sim = process_slide("synthetic_tumor", "Tumor", feats, coords, ps, protos, names, colors,
                                   ann, stats, thumb, dsx, dsy, args, footer="self-test")
    assert (idx == cls).mean() > 0.98, (idx == cls).mean()
    r = {x["class"]: x for x in rows}
    assert r["tumor"]["pct_tumor"] > 95 and r["tumor"]["n_non_tumor"] <= 2
    assert sum(x["n_all"] for x in rows) == len(coords)
    assert abs(sum(x["pct_tumor"] for x in rows) - 100) < 1e-9
    # ambiguity: margin larger than any gap -> every patch ambiguous
    amb = assign_classes(sim, min_margin=10.0)
    assert (amb == len(names)).all()
    # normal slide: no contour, no inside/outside columns
    args.exp_name = "normal"
    rows_n, _, _ = process_slide("synthetic_normal", "Normal", feats[~lesion], coords[~lesion], ps, protos,
                                 names, colors, None, None, thumb, dsx, dsy, args, footer="self-test")
    assert "pct_tumor" not in rows_n[0]
    print(f"  assignment accuracy on the synthetic slide {(idx == cls).mean():.3f}; "
          f"inside the lesion: tumor {r['tumor']['pct_tumor']:.1f}%")

    # ---- 11 classes: 6 normal + 5 tumor, with groups --------------------------------
    normal = ["lymphocytes", "germinal_center", "sinus_histiocytes", "adipose_tissue",
              "fibrous_stroma_vessels", "artifact_background"]
    tumor = ["solid_carcinoma_nests", "glandular_carcinoma", "discohesive_lobular_carcinoma",
             "isolated_tumor_cells", "desmoplastic_tumor_stroma"]
    spec11 = {"classes": {n: f"text for {n}" for n in normal + tumor},
              "groups": {"normal": normal, "tumor": tumor}}
    p3 = os.path.join(tmp, "c.json")
    json.dump(spec11, open(p3, "w"))
    n11, _, c11, g11 = load_class_spec(p3)
    assert len(n11) == 11 and len(set(c11)) == 11
    assert c11[:6] == FAMILY_PALETTES["normal"][:6] and c11[6:] == FAMILY_PALETTES["tumor"][:5]
    assert g11 == ["normal"] * 6 + ["tumor"] * 5
    # same classes without groups: extended palette, still 11 distinct colors
    json.dump({"classes": spec11["classes"]}, open(p3, "w"))
    _, _, c11b, g11b = load_class_spec(p3)
    assert g11b is None and len(set(c11b)) == 11 and c11b == PALETTE_EXT[:11]
    # errors: unknown / ungrouped class, too many classes without colors
    for bad in ({"classes": {"a": "x", "b": "y"}, "groups": {"normal": ["a", "zz"]}},
                {"classes": {"a": "x", "b": "y"}, "groups": {"normal": ["a"]}},
                {f"c{i}": "x" for i in range(13)}):
        json.dump(bad, open(p3, "w"))
        try:
            load_class_spec(p3)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    protos11 = np.linalg.qr(rng.standard_normal((d, 11)))[0].astype(np.float32)
    cls11 = np.zeros(len(coords), int)
    cls11[gx.ravel() < 5] = 3                                         # fat
    cls11[(gx.ravel() >= 5) & (gx.ravel() < 8)] = 4                   # stroma
    cls11[((gx.ravel() - 18) ** 2 + (gy.ravel() - 6) ** 2) < 9] = 1   # germinal centre
    cls11[(gx.ravel() == 28) & (gy.ravel() < 6)] = 2                  # sinus
    cls11[lesion] = 6                                                 # solid nests
    cls11[lesion & (gx.ravel() == 23)] = 9                            # isolated cells at the edge
    cls11[(gx.ravel() == 12) & (gy.ravel() == 15)] = 9                # an ITC-like false positive
    f11 = protos11.T[cls11] + 0.15 * rng.standard_normal((len(coords), d))
    f11 = (f11 / np.linalg.norm(f11, axis=1, keepdims=True)).astype(np.float32)
    args.exp_name = "11classes"
    rows11, idx11, _ = process_slide("synthetic_tumor", "Tumor", f11, coords, ps, protos11, n11, c11,
                                     ann, stats, thumb, dsx, dsy, args, footer="self-test", groups=g11)
    assert (idx11 == cls11).mean() > 0.98
    sub = {x["class"]: x for x in rows11}
    assert sub["[tumor]"]["pct_tumor"] > 95 and sub["[tumor]"]["n_non_tumor"] == 1
    assert sub["[normal]"]["n_all"] + sub["[tumor]"]["n_all"] == len(coords)
    assert all(x["group"] in ("normal", "tumor") for x in rows11)
    print(f"  11 classes in 2 groups: accuracy {(idx11 == cls11).mean():.3f}; "
          f"tumor group inside the lesion {sub['[tumor]']['pct_tumor']:.1f}%")
    print(f"self_test: OK ({out_dir})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide_ids", nargs="+")
    p.add_argument("--class_json")
    p.add_argument("--wsi_dir")
    p.add_argument("--xml_dir")
    p.add_argument("--folder", default=None,
                   help="MIL-format root (<folder>/<project>/<encoder>/<slide>.npy, CONCH-projected). "
                        "If omitted, the raw .h5 features are projected with CONCH.")
    p.add_argument("--h5_dir", default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                                       "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--csv_path", default=None)
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--checkpoint_path",
                   default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--min_margin", type=float, default=0.0,
                   help="Cosine gap between best and second class below which a patch is 'ambiguous'.")
    p.add_argument("--alpha", type=float, default=0.75)
    p.add_argument("--show_gt", action="store_true",
                   help="Also draw a separate ground-truth panel (the class map always carries the "
                        "tumor contour on tumor slides).")
    p.add_argument("--thumb_max_dim", type=int, default=3000)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--out_dir", default="./visualizations/zs_classes")
    p.add_argument("--exp_name", default="")
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test(args.out_dir if args.out_dir != "./visualizations/zs_classes" else None)
        return
    missing = [a for a in ("slide_ids", "class_json", "wsi_dir", "xml_dir") if not getattr(args, a)]
    if missing:
        p.error("required: " + ", ".join("--" + m for m in missing) + " (or --self_test)")
    main_run(args)


if __name__ == "__main__":
    main()
