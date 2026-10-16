"""
utils/witness_rate.py

Patch-level tumor annotation and per-slide witness rate for CAMELYON16-style data,
with the patch footprint taken from the h5 file rather than assumed.

THE BUG THIS MODULE EXISTS TO PREVENT
-------------------------------------
TRIDENT stores the patch side in level-0 pixels as an attribute of the "coords"
DATASET. Reading it from the file root (h5f.attrs.get("patch_size_level0", 512)) falls
back to 512, while CAMELYON16 patches extracted at 20x / 512 px span 1024 level-0 px
(slides are scanned at 40x). Every patch was then treated as its own top-left quarter:
centres were tested at (x+256, y+256) instead of (x+512, y+512), and coverage was
measured over 25% of the patch. read_patch_size_level0() reads the right attribute and
cross-checks it against the coordinate grid.

WHAT IS PROVIDED
----------------
Patch geometry
    read_patch_size_level0(h5f, coords)     coords attrs -> root attrs -> grid pitch,
                                            with a consistency check against the grid
    load_patch_grid(h5_path)                coords + patch side (+ features if asked)

Annotations
    parse_camelyon16_xml(xml_path)          ASAP XML -> Annotations(lesions, exclusions).
                                            Groups "_0"/"_1" are tumor, "_2" are
                                            non-tumor holes inside a tumor region and
                                            are SUBTRACTED (visu_ABMIL.parse_camelyon16_xml
                                            counts them as tumor).
    read_mask(path, level)                  CAMELYON16 tumor mask .tif -> boolean array
                                            at a pyramid level + its downsample factor

Patch labels (one value per patch)
    patch_occupancy_from_polygons(...)      EXACT fraction of each patch's area inside
                                            tumor (polygon clipping, no sampling, so an
                                            isolated-tumor-cell cluster is never missed)
    patch_centre_in_polygons(...)           centre rule, exclusions removed
    patch_occupancy_from_mask(...)          same quantities from a raster mask
    patch_centre_in_mask(...)
    lesion_geometry(ann, mpp)               per-lesion area, diameters, AJCC stage

Per-slide witness rate
    slide_witness_stats(occupancy, centre)  every definition side by side (below)
    compute_slide(h5_path, xml_path=..., mask_path=...)
                                            one call: stats dict + per-patch DataFrame

WITNESS-RATE DEFINITIONS (all in percent of the slide's TISSUE patches, i.e. of what
TRIDENT extracted, which is what the MIL model sees)
    witness_rate_centre_pct  patches whose CENTRE is in tumor. Same definition as the
                             historical tumor_patch_pct, now on the correct footprint.
    witness_rate_occ_pct     patches at least `min_occupancy` covered (default 0.5).
    touched_pct              patches with ANY tumor (occupancy > 0): the upper bound on
                             what can contain a witness.
    tumor_area_pct           sum of occupancies / N: tumor AREA as a share of the tissue
                             area, the definition least sensitive to where the patch grid
                             happens to fall. Recommended for the x-axis.

Run `python -m utils.witness_rate` for the self-test (no data needed).
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from matplotlib.path import Path as MplPath

TUMOR_GROUPS = ("_0", "_1", "tumor", "metastases")
EXCLUSION_GROUPS = ("_2", "exclusion", "none")
# AJCC / CAMELYON size classes, greatest dimension in mm
STAGE_EDGES = {"ITC": (0.0, 0.2), "micro": (0.2, 2.0), "macro": (2.0, np.inf)}
DEFAULT_MPP_LEVEL0 = 0.243          # um/px, CAMELYON16 3DHistech; Hamamatsu is ~0.226


# =====================================================================================
# Patch geometry
# =====================================================================================

def grid_pitch(coords):
    """Smallest positive gap between distinct x (or y) coordinates, in level-0 px.

    With non-overlapping patches this is exactly the patch side at level 0.
    """
    c = np.asarray(coords, float)
    best = np.inf
    for axis in (0, 1):
        u = np.unique(c[:, axis])
        d = np.diff(u)
        d = d[d > 0]
        if len(d):
            best = min(best, float(d.min()))
    return best if np.isfinite(best) else np.nan


def read_patch_size_level0(h5f, coords=None, check_grid=True):
    """Patch side in level-0 pixels for an open TRIDENT h5 file.

    Order of trust: f["coords"].attrs, then f.attrs, then the coordinate grid pitch.
    If an attribute disagrees with a well-defined grid pitch the GRID wins and a
    warning is printed: patches cannot overlap less than they are spaced, so a recorded
    size smaller than the spacing is certainly wrong.
    """
    if coords is None:
        coords = h5f["coords"][:]
    pitch = grid_pitch(coords) if check_grid and len(coords) > 1 else np.nan

    recorded = None
    for obj in (h5f["coords"], h5f):
        if "patch_size_level0" in obj.attrs:
            recorded = int(obj.attrs["patch_size_level0"])
            break
    if recorded is None:
        if np.isfinite(pitch):
            return int(round(pitch))
        raise ValueError("patch_size_level0 not stored and cannot be inferred from a "
                         "single-patch grid")
    if np.isfinite(pitch) and recorded < pitch - 0.5:
        print(f"WARNING: patch_size_level0 attribute = {recorded} but patches are "
              f"{pitch:g} level-0 px apart; using {pitch:g}.")
        return int(round(pitch))
    return recorded


@dataclass
class PatchGrid:
    coords: np.ndarray            # (N, 2) level-0 top-left corners
    patch_size: int               # level-0 side
    features: np.ndarray = None

    @property
    def centres(self):
        return self.coords + self.patch_size / 2.0

    def __len__(self):
        return len(self.coords)


def load_patch_grid(h5_path, with_features=False):
    import h5py
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:].astype(np.float64)
        ps = read_patch_size_level0(f, coords)
        feats = f["features"][:] if with_features else None
    return PatchGrid(coords, ps, feats)


# =====================================================================================
# Polygon geometry
# =====================================================================================

def polygon_area(poly):
    """Unsigned shoelace area, squared input units."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))


def _clip_halfplane(P, axis, bound, keep_greater):
    """One Sutherland-Hodgman pass against x/y >= bound (or <= bound). Vectorised."""
    if len(P) == 0:
        return P
    Q = np.roll(P, -1, axis=0)
    inP = P[:, axis] >= bound if keep_greater else P[:, axis] <= bound
    inQ = Q[:, axis] >= bound if keep_greater else Q[:, axis] <= bound
    cross = inP != inQ
    denom = Q[:, axis] - P[:, axis]
    t = np.divide(bound - P[:, axis], denom, out=np.zeros(len(P)), where=cross)
    I = P + t[:, None] * (Q - P)
    I[:, axis] = np.where(cross, bound, I[:, axis])      # exact on the boundary
    pts = np.stack([I, Q], axis=1)                        # emit I then Q per edge
    keep = np.stack([cross, inQ], axis=1)
    return pts[keep]


def clip_polygon_to_box(poly, x0, y0, x1, y1):
    """Polygon ∩ axis-aligned box (Sutherland-Hodgman). The clip window is convex, so
    the result's area is exact for any simple subject polygon, concave included."""
    P = np.asarray(poly, float)
    for axis, bound, keep in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        P = _clip_halfplane(P, axis, bound, keep)
        if len(P) == 0:
            break
    return P


def _convex_hull(points):
    p = np.unique(np.asarray(points, float), axis=0)
    if len(p) <= 2:
        return p
    p = p[np.lexsort((p[:, 1], p[:, 0]))]
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for pt in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    for pt in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    return np.asarray(lower[:-1] + upper[:-1])


def max_feret_diameter(poly):
    """Greatest dimension (AJCC definition) = max pairwise distance on the convex hull."""
    hull = _convex_hull(poly)
    if len(hull) < 2:
        return 0.0
    d2 = ((hull[:, None, :] - hull[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d2.max()))


# =====================================================================================
# Annotations
# =====================================================================================

@dataclass
class Annotations:
    lesions: list = field(default_factory=list)      # list of (M, 2) level-0 polygons
    exclusions: list = field(default_factory=list)   # holes inside tumor regions
    groups: list = field(default_factory=list)       # group name per lesion

    def __len__(self):
        return len(self.lesions)


def parse_camelyon16_xml(xml_path):
    """CAMELYON16 ASAP XML -> Annotations. Unknown group names count as tumor (the
    conservative choice: CAMELYON16 is not perfectly consistent about group names)."""
    root = ET.parse(xml_path).getroot()
    ann = Annotations()
    for a in root.iter("Annotation"):
        ce = a.find("Coordinates")
        if ce is None:
            continue
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in ce.iter("Coordinate")]
        if len(pts) < 3:
            continue
        poly = np.asarray(pts, np.float64)
        group = (a.get("PartOfGroup") or "").strip()
        if group.lower() in EXCLUSION_GROUPS:
            ann.exclusions.append(poly)
        else:
            ann.lesions.append(poly)
            ann.groups.append(group)
    return ann


def _boxes_overlapping(coords, ps, lo, hi):
    """Indices of patches whose box intersects the bbox [lo, hi]."""
    c = coords
    m = ((c[:, 0] < hi[0]) & (c[:, 0] + ps > lo[0]) &
         (c[:, 1] < hi[1]) & (c[:, 1] + ps > lo[1]))
    return np.nonzero(m)[0]


def _polygon_patch_areas(coords, ps, poly):
    """{patch index: area of poly inside that patch} for every patch it touches."""
    out = {}
    lo, hi = poly.min(0), poly.max(0)
    for i in _boxes_overlapping(coords, ps, lo, hi):
        x0, y0 = coords[i]
        a = polygon_area(clip_polygon_to_box(poly, x0, y0, x0 + ps, y0 + ps))
        if a > 0:
            out[i] = a
    return out


def patch_occupancy_from_polygons(coords, patch_size, ann, per_lesion=False):
    """Fraction of each patch's AREA that is tumor, exact (polygon clipping).

    Exclusion polygons are subtracted. Overlapping tumor polygons are capped at 1.
    per_lesion=True also returns, for each lesion, (patch indices, occupancies), with
    each exclusion subtracted from the lesion whose bounding box contains it.
    """
    coords = np.asarray(coords, float)
    ps = float(patch_size)
    pa = ps * ps
    occ = np.zeros(len(coords))
    les_maps = []
    for poly in ann.lesions:
        m = _polygon_patch_areas(coords, ps, poly)
        les_maps.append(m)
        for i, a in m.items():
            occ[i] += a / pa
    excl_maps = [_polygon_patch_areas(coords, ps, ex) for ex in ann.exclusions]
    for m in excl_maps:
        for i, a in m.items():
            occ[i] -= a / pa
    occ = np.clip(occ, 0.0, 1.0)
    if not per_lesion:
        return occ

    per = []
    boxes = [(p.min(0), p.max(0)) for p in ann.lesions]
    owner = []
    for ex in ann.exclusions:
        lo, hi = ex.min(0), ex.max(0)
        own = [j for j, (l, h) in enumerate(boxes) if np.all(lo >= l) and np.all(hi <= h)]
        owner.append(own[0] if own else None)
    for j, m in enumerate(les_maps):
        m = dict(m)
        for k, em in enumerate(excl_maps):
            if owner[k] == j:
                for i, a in em.items():
                    if i in m:
                        m[i] -= a
        idx = np.array(sorted(i for i in m if m[i] > 0), dtype=int)
        per.append((idx, np.clip(np.array([m[i] / pa for i in idx]), 0, 1)))
    return occ, per


def patch_centre_in_polygons(coords, patch_size, ann):
    """True where the patch CENTRE lies in a tumor polygon and in no exclusion."""
    centres = np.asarray(coords, float) + patch_size / 2.0
    inside = np.zeros(len(centres), bool)
    for poly in ann.lesions:
        inside |= MplPath(poly).contains_points(centres)
    for ex in ann.exclusions:
        inside &= ~MplPath(ex).contains_points(centres)
    return inside


def lesion_geometry(ann, mpp_level0=DEFAULT_MPP_LEVEL0):
    """Per lesion: area (exclusions inside its bbox subtracted), equivalent-circle and
    greatest-dimension diameters in mm, AJCC stage."""
    import pandas as pd
    rows = []
    px_mm = mpp_level0 * 1e-3
    for j, poly in enumerate(ann.lesions):
        lo, hi = poly.min(0), poly.max(0)
        area = polygon_area(poly)
        for ex in ann.exclusions:
            if np.all(ex.min(0) >= lo) and np.all(ex.max(0) <= hi):
                area -= polygon_area(ex)
        area_mm2 = max(area, 0.0) * px_mm ** 2
        dmax = max_feret_diameter(poly) * px_mm
        rows.append({"lesion_id": j, "group": ann.groups[j] if j < len(ann.groups) else "",
                     "area_mm2": area_mm2, "diam_eq_mm": 2 * np.sqrt(area_mm2 / np.pi),
                     "diam_max_mm": dmax, "stage": stage_of(dmax)})
    return pd.DataFrame(rows)


def stage_of(diam_mm):
    for name, (lo, hi) in STAGE_EDGES.items():
        if lo <= diam_mm < hi:
            return name
    return "macro"


# =====================================================================================
# Raster masks (CAMELYON16 also ships level-0 tumor masks as pyramidal .tif)
# =====================================================================================

def read_mask(path, level=4):
    """Tumor mask at a pyramid level -> (bool array [H, W], downsample to level 0).

    Uses OpenSlide (module load openslide on the cluster). Level 4 (16x) keeps a
    CAMELYON16 mask around 6000 x 6000 px; a 1024-px patch is then 64 x 64 mask px.
    Any non-zero value counts as tumor.
    """
    import openslide
    s = openslide.OpenSlide(path)
    level = min(level, s.level_count - 1)
    w, h = s.level_dimensions[level]
    ds = float(s.level_downsamples[level])
    arr = np.asarray(s.read_region((0, 0), level, (w, h)))[..., 0] > 0
    s.close()
    return arr, ds


def _integral(mask):
    ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(mask.astype(np.int64), 0), 1)
    return ii


def patch_occupancy_from_mask(coords, patch_size, mask, downsample):
    """Fraction of each patch's mask pixels that are tumor (box sums, O(1) per patch).
    Patch parts falling outside the mask count as non-tumor."""
    ii = _integral(mask)
    H, W = mask.shape
    c = np.asarray(coords, float) / downsample
    side = patch_size / downsample
    x0 = np.clip(np.floor(c[:, 0]).astype(int), 0, W)
    y0 = np.clip(np.floor(c[:, 1]).astype(int), 0, H)
    x1 = np.clip(np.ceil(c[:, 0] + side).astype(int), 0, W)
    y1 = np.clip(np.ceil(c[:, 1] + side).astype(int), 0, H)
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    full = np.maximum(np.ceil(c[:, 0] + side).astype(int) - np.floor(c[:, 0]).astype(int), 1) * \
           np.maximum(np.ceil(c[:, 1] + side).astype(int) - np.floor(c[:, 1]).astype(int), 1)
    return np.clip(s / full, 0.0, 1.0)


def patch_centre_in_mask(coords, patch_size, mask, downsample):
    H, W = mask.shape
    cen = (np.asarray(coords, float) + patch_size / 2.0) / downsample
    x = np.clip(cen[:, 0].astype(int), 0, W - 1)
    y = np.clip(cen[:, 1].astype(int), 0, H - 1)
    inb = (cen[:, 0] >= 0) & (cen[:, 0] < W) & (cen[:, 1] >= 0) & (cen[:, 1] < H)
    return mask[y, x] & inb


# =====================================================================================
# Per-slide witness rate
# =====================================================================================

def slide_witness_stats(occupancy, centre, min_occupancy=0.5):
    """Every witness-rate definition for one slide (see module docstring)."""
    occ = np.asarray(occupancy, float)
    cen = np.asarray(centre, bool)
    n = len(occ)
    if n == 0:
        return {"n_patches": 0}
    pct = lambda k: 100.0 * k / n
    n_occ = int((occ >= min_occupancy).sum())
    return {
        "n_patches": n,
        "n_tumor_centre": int(cen.sum()),
        "witness_rate_centre_pct": pct(cen.sum()),
        "n_tumor_occ": n_occ,
        "witness_rate_occ_pct": pct(n_occ),
        "min_occupancy": min_occupancy,
        "n_touched": int((occ > 0).sum()),
        "touched_pct": pct((occ > 0).sum()),
        "tumor_area_pct": 100.0 * float(occ.sum()) / n,
    }


def compute_slide(h5_path, xml_path=None, mask_path=None, mask_level=4,
                  min_occupancy=0.5, mpp_level0=DEFAULT_MPP_LEVEL0, with_lesions=False):
    """One slide end to end.

    Annotation source: xml_path (polygons, exact) or mask_path (raster); neither means
    a normal slide (everything 0). Returns (stats dict, per-patch DataFrame[, lesions]).
    """
    import pandas as pd
    grid = load_patch_grid(h5_path)
    n = len(grid)
    lesions = None
    if xml_path is not None:
        ann = parse_camelyon16_xml(xml_path)
        if with_lesions:
            occ, per = patch_occupancy_from_polygons(grid.coords, grid.patch_size, ann, True)
            lesions = lesion_geometry(ann, mpp_level0)
            if len(lesions):
                lesions["n_patches_touched"] = [len(i) for i, _ in per]
                lesions["n_patches_occ"] = [int((o >= min_occupancy).sum()) for _, o in per]
        else:
            occ = patch_occupancy_from_polygons(grid.coords, grid.patch_size, ann)
        cen = patch_centre_in_polygons(grid.coords, grid.patch_size, ann)
        source, n_les = "xml", len(ann)
    elif mask_path is not None:
        mask, ds = read_mask(mask_path, mask_level)
        occ = patch_occupancy_from_mask(grid.coords, grid.patch_size, mask, ds)
        cen = patch_centre_in_mask(grid.coords, grid.patch_size, mask, ds)
        source, n_les = f"mask(level {mask_level})", np.nan
    else:
        occ, cen = np.zeros(n), np.zeros(n, bool)
        source, n_les = "none", 0

    stats = slide_witness_stats(occ, cen, min_occupancy)
    stats.update({"patch_size_level0": grid.patch_size, "annotation_source": source,
                  "n_lesions": n_les})
    patches = pd.DataFrame({"x": grid.coords[:, 0], "y": grid.coords[:, 1],
                            "occupancy": occ, "tumor_centre": cen,
                            "tumor_occ": occ >= min_occupancy})
    return (stats, patches, lesions) if with_lesions else (stats, patches)


# =====================================================================================
# Self-test (no dataset needed)
# =====================================================================================

def self_test():
    import os
    import tempfile
    rng = np.random.default_rng(0)
    print("self_test: utils.witness_rate")

    # --- clipping is exact: square, concave L, random star vs dense sampling --------
    sq = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)
    assert abs(polygon_area(clip_polygon_to_box(sq, 5, 5, 20, 20)) - 25) < 1e-9
    assert len(clip_polygon_to_box(sq, 20, 20, 30, 30)) == 0
    L = np.array([[0, 0], [10, 0], [10, 2], [2, 2], [2, 10], [0, 10]], float)
    assert abs(polygon_area(clip_polygon_to_box(L, 1, 1, 5, 5)) - 7) < 1e-9   # 4x1 + 1x3
    th = np.linspace(0, 2 * np.pi, 41)[:-1]
    r = 300 + 150 * rng.random(40)
    star = np.column_stack([500 + r * np.cos(th), 500 + r * np.sin(th)])
    box = (600.0, 300.0, 1000.0, 700.0)      # straddles the star boundary
    xs = np.linspace(box[0], box[2], 801)[:-1] + 0.25
    ys = np.linspace(box[1], box[3], 801)[:-1] + 0.25
    gx, gy = np.meshgrid(xs, ys)
    mc = MplPath(star).contains_points(np.column_stack([gx.ravel(), gy.ravel()])).mean()
    exact = polygon_area(clip_polygon_to_box(star, *box)) / ((box[2] - box[0]) * (box[3] - box[1]))
    assert 0.05 < exact < 0.95 and abs(exact - mc) < 0.005, (exact, mc)
    print(f"  polygon clipping exact (concave L, random star {exact:.4f} vs sampled {mc:.4f})")

    # --- patch geometry: the actual bug -------------------------------------------
    ps = 1024
    gxp, gyp = np.meshgrid(np.arange(6), np.arange(6))
    coords = np.column_stack([gxp.ravel(), gyp.ravel()]).astype(float) * ps
    assert grid_pitch(coords) == 1024
    try:
        import h5py
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "a.h5")
            with h5py.File(p, "w") as f:                       # TRIDENT layout
                d = f.create_dataset("coords", data=coords)
                d.attrs["patch_size_level0"] = 1024
                d.attrs["patch_size"] = 512
            with h5py.File(p, "r") as f:
                assert read_patch_size_level0(f) == 1024
                assert f.attrs.get("patch_size_level0", 512) == 512   # the old read
            with h5py.File(p, "w") as f:                       # no attribute anywhere
                f.create_dataset("coords", data=coords)
            assert load_patch_grid(p).patch_size == 1024
            with h5py.File(p, "w") as f:                       # wrong attribute
                f.create_dataset("coords", data=coords).attrs["patch_size_level0"] = 512
            with h5py.File(p, "r") as f:
                assert read_patch_size_level0(f) == 1024
        print("  patch size: coords attrs 1024 (root fallback would give 512); grid "
              "fallback and wrong-attribute override both give 1024")
    except ImportError:
        print("  (h5py not installed: h5 reading not exercised)")

    # --- a lesion in the BOTTOM-RIGHT quarter of patch (2, 2): the old read missed it
    c = np.array([2 * ps + 800, 2 * ps + 800])
    tiny = c + 60 * np.column_stack([np.cos(th), np.sin(th)])        # ~30 um across
    ann = Annotations([tiny], [], ["_0"])
    occ_new = patch_occupancy_from_polygons(coords, ps, ann)
    occ_old = patch_occupancy_from_polygons(coords, 512, ann)
    host = np.where((coords == [2 * ps, 2 * ps]).all(1))[0][0]
    assert occ_new[host] > 0 and (occ_new > 0).sum() == 1
    assert occ_old.sum() == 0
    expect = polygon_area(tiny) / ps ** 2
    assert abs(occ_new[host] - expect) < 1e-9
    print(f"  ITC-size lesion in a patch's bottom-right quarter: occupancy {occ_new[host]:.4f}"
          " with 1024 px, invisible with 512 px")

    # --- lesion straddling 4 patches; exclusion hole; centre rule ------------------
    sq4 = np.array([[1.55, 1.55], [2.45, 1.55], [2.45, 2.45], [1.55, 2.45]]) * ps
    hole = np.array([[2.0, 2.0], [2.25, 2.0], [2.25, 2.25], [2.0, 2.25]]) * ps
    ann = Annotations([sq4], [hole], ["_0"])
    occ, per = patch_occupancy_from_polygons(coords, ps, ann, per_lesion=True)
    area = 0.9 ** 2 - 1 / 16                             # square minus hole, in patches
    assert (occ > 0).sum() == 4 and abs(occ.sum() - area) < 1e-9, occ.sum()
    assert abs(per[0][1].sum() - occ.sum()) < 1e-9
    cen = patch_centre_in_polygons(coords, ps, ann)
    assert cen.sum() == 0                                    # no centre inside: straddler
    st = slide_witness_stats(occ, cen)
    assert st["witness_rate_centre_pct"] == 0 and st["n_touched"] == 4
    assert abs(st["tumor_area_pct"] - 100 * area / 36) < 1e-9
    print(f"  4-patch straddler with a hole: centre rule 0%, touched {st['touched_pct']:.1f}%, "
          f"area {st['tumor_area_pct']:.2f}% -- why tumor_area_pct is the robust x-axis")

    # --- mask path agrees with polygons ------------------------------------------
    ds = 16.0
    H = W = int(6 * ps / ds)
    yy, xx = np.mgrid[0:H, 0:W]
    pix = np.column_stack([(xx.ravel() + 0.5) * ds, (yy.ravel() + 0.5) * ds])
    big = np.array([[0.3, 0.6], [4.2, 0.4], [4.8, 3.9], [1.1, 4.6]]) * ps
    mask = MplPath(big).contains_points(pix).reshape(H, W)
    o_poly = patch_occupancy_from_polygons(coords, ps, Annotations([big], [], ["_0"]))
    o_mask = patch_occupancy_from_mask(coords, ps, mask, ds)
    assert np.abs(o_poly - o_mask).max() < 0.03, np.abs(o_poly - o_mask).max()
    c_poly = patch_centre_in_polygons(coords, ps, Annotations([big], [], ["_0"]))
    c_mask = patch_centre_in_mask(coords, ps, mask, ds)
    assert (c_poly == c_mask).mean() > 0.95
    print(f"  mask vs polygon occupancy: max |diff| {np.abs(o_poly - o_mask).max():.3f} "
          f"at 16x downsample")

    # --- XML round trip with groups ----------------------------------------------
    xml = ("<ASAP_Annotations><Annotations>"
           + "".join(f'<Annotation Name="A{i}" PartOfGroup="{g}"><Coordinates>'
                     + "".join(f'<Coordinate Order="{k}" X="{x}" Y="{y}"/>'
                               for k, (x, y) in enumerate(poly))
                     + "</Coordinates></Annotation>"
                     for i, (g, poly) in enumerate([("_0", sq4), ("_2", hole), ("_1", tiny)]))
           + "</Annotations></ASAP_Annotations>")
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "a.xml")
        open(p, "w").write(xml)
        a = parse_camelyon16_xml(p)
    assert len(a.lesions) == 2 and len(a.exclusions) == 1
    geo = lesion_geometry(a, 0.243)
    assert list(geo["stage"]) == ["micro", "ITC"]
    print(f"  XML: 2 lesions + 1 exclusion; stages {list(geo['stage'])}")
    print("self_test: OK")


if __name__ == "__main__":
    self_test()
