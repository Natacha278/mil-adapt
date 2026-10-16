"""
compare_aggregation_vs_witness_rate.py

Slide-level performance of different aggregation methods as a function of witness
rate, on CAMELYON16, all evaluated on the SAME slides: for every few-shot seed, that
seed's evaluation split (its k + k support slides are excluded for every method, also
for the training-free ones).

METHODS (any combination)
  MI-Zero top-K   training-free: per class, the mean of the K highest patch-prompt cosine
                  similarities; label = argmax; score = pooled Tumor - pooled Normal.
                  --mizero_k 1 5 50 0.01  (K < 1 = fraction of the slide's patches).
                  Computed with numpy from the MIL-format .npy features (already projected
                  CONCH embeddings), no GPU.
  MIL-Adapter     adapter + ABMIL trained per seed exactly as main.py (--train_miladapter,
                  GPU), score = P(Tumor).
  precomputed     any per-slide CSV with columns slide_id, gt_label, pred_label, score
                  [, seed] via --per_slide "LABEL=path" (e.g. the outputs of
                  analyze_tumor_ratio_miladapter.py / analyze_tumor_ratio_mizero.py).
                  A file with a `seed` column must contain exactly each seed's eval split.

WITNESS RATE AND STAGE per tumor slide, first source available:
  1. columns tumor_area_pct and lesion_stage in --csv_path (add_tumor_pct_to_csv.py);
  2. --lesion_csv, a per-lesion table (WSI, area_mm2, diam_max_mm, stage), with the
     tissue area = n_patches x (patch side x mpp)^2;
  3. computed from --h5_dir and --xml_dir with utils/witness_rate.py.
  The x-axis is tumor AREA as % of tissue (grid-independent); stage = largest lesion
  (ITC < 0.2 mm <= micro < 2 mm <= macro, greatest dimension).

METRICS per seed, then mean +- SD across seeds
  detection rate   tumor slides predicted Tumor (slide sensitivity), per bin / stage
  specificity      normal slides predicted Normal
  AUC              threshold-free: tumor slides of the bin / stage vs ALL normal eval
                   slides (mean placement value, as utils/tumor_ratio_metrics.py)
  overall          accuracy, balanced accuracy, AUC

  Zero-shot labels use the argmax of the pooled similarities, whose operating point is
  not calibrated (MI-Zero often calls almost everything Normal). --calibrate_zero_shot
  sets, per seed, the zero-shot threshold at the --calib_quantile of the seed's SUPPORT
  normal slides' scores (legitimate: those slides are labelled in the few-shot setting).
  AUC does not depend on the threshold at all.

FIGURES
  <out_prefix>_by_witness_rate.png   detection rate (bars) + AUC (lines) per bin
  <out_prefix>_by_stage.png          same for Normal | ITC | micro | macro
CSVs: _per_slide.csv, _per_seed.csv, _by_bin.csv, _by_stage.csv

    # from existing outputs (CPU, seconds)
    python compare_aggregation_vs_witness_rate.py \
        --per_slide "MIL-Adapter (TaskRes+ABMIL)=analysis/analysis_auc/CAMELYON16_CONCH_TaskRes_ABMIL_k16_per_slide.csv" \
                    "MI-Zero top-1=analysis/analysis_mizero/CAMELYON16_MIZERO_logit_k1_per_slide.csv" \
                    "MI-Zero top-50=analysis/analysis_mizero/CAMELYON16_MIZERO_logit_k50_per_slide.csv" \
        --lesion_csv analysis/analysis_lesion_size_descr/CAMELYON16_CONCH_zs_per_lesion.csv \
        --out_prefix analysis/analysis_compare/descr_k16

    # computing everything (GPU for MIL-Adapter)
    python compare_aggregation_vs_witness_rate.py \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --mizero_k 1 5 50 --train_miladapter --adapter TaskRes \
        --xml_dir $SCRATCH/camelyon16/annotations --out_prefix analysis/analysis_compare/descr_k16
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import tumor_ratio_metrics as TRM           # noqa: E402

CLASSES = ["Normal", "Tumor"]
WR_BINS = [0.0, 0.1, 0.5, 2.0, 10.0, 100.0]
WR_LABELS = [f"{lo:g}–{hi:g}%" for lo, hi in zip(WR_BINS[:-1], WR_BINS[1:])]
STAGES = ["ITC", "micro", "macro"]
STAGE_EDGES = [0.2, 2.0]


def stage_of(d):
    return "ITC" if d < STAGE_EDGES[0] else "micro" if d < STAGE_EDGES[1] else "macro"


# =====================================================================================
# Witness rate / stage
# =====================================================================================

def witness_from_lesion_csv(meta, lesion_csv, patch_size_level0, mpp):
    les = pd.read_csv(lesion_csv)
    tissue_mm2 = meta.set_index("WSI")["n_patches"] * (patch_size_level0 * mpp * 1e-3) ** 2
    g = les.groupby("WSI")
    out = pd.DataFrame({"tumor_area_mm2": g.area_mm2.sum(),
                        "max_lesion_diam_mm": g.diam_max_mm.max()})
    out["tumor_area_pct"] = 100 * out.tumor_area_mm2 / tissue_mm2.reindex(out.index)
    out["lesion_stage"] = out.max_lesion_diam_mm.map(stage_of)
    return out[["tumor_area_pct", "lesion_stage", "max_lesion_diam_mm"]]


def witness_from_annotations(meta, h5_dir, xml_dir, mpp):
    from utils import witness_rate as W
    rows = {}
    for w in meta.loc[meta.GT == "Tumor", "WSI"]:
        h5, xml = os.path.join(h5_dir, f"{w}.h5"), os.path.join(xml_dir, f"{w}.xml")
        if not (os.path.exists(h5) and os.path.exists(xml)):
            continue
        st, _p, les = W.compute_slide(h5, xml_path=xml, mpp_level0=mpp, with_lesions=True)
        d = float(les.diam_max_mm.max()) if les is not None and len(les) else np.nan
        rows[w] = {"tumor_area_pct": st["tumor_area_pct"], "max_lesion_diam_mm": d,
                   "lesion_stage": stage_of(d) if np.isfinite(d) else None}
    return pd.DataFrame.from_dict(rows, orient="index")


# =====================================================================================
# Method sources
# =====================================================================================

def mizero_scores(X, text_prototypes, k):
    """MI-Zero top-K on projected, L2-normalised features (numpy)."""
    P = np.asarray(text_prototypes, float)
    if P.shape[0] < P.shape[1]:
        P = P.T
    P = P / np.linalg.norm(P, axis=0, keepdims=True)
    out = []
    for H in X:
        H = np.asarray(H, float)
        sim = (H / np.linalg.norm(H, axis=1, keepdims=True)) @ P
        kk = int(np.clip(np.ceil(k * len(sim)) if 0 < k < 1 else k, 1, len(sim)))
        pooled = -np.sort(-sim, axis=0)[:kk].mean(axis=0)
        out.append((float(pooled[1] - pooled[0]), CLASSES[int(np.argmax(pooled))]))
    return out


def train_miladapter_scores(X, Y, WSI, seeds, k_shots, args):
    """{seed: {wsi: (P(Tumor), pred)}} for each seed's eval slides."""
    import torch
    from utils.trainer import train_model
    from utils.utils import set_random_seeds, fewshot_sampling
    from visu_ABMIL import build_model, patch_known_bugs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_np = np.load(args.text_prototypes)
    out = {}
    for seed in seeds:
        set_random_seeds(seed_value=seed)
        tr, va, trl, _ = fewshot_sampling(X=X, Y=Y, k_shots=k_shots, seed=seed)
        idx = {id(x): i for i, x in enumerate(X)}
        model = patch_known_bugs(build_model(args.adapter, text_np, "ABMIL", args.init, tr, trl)
                                 .to(device), args.adapter, "ABMIL")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        train_model(model, opt, torch.nn.CrossEntropyLoss(reduction="sum"), sch, tr, trl,
                    args.epochs)
        model.eval()
        res = {}
        with torch.no_grad():
            for v in va:
                p = float(model(torch.as_tensor(v, dtype=torch.float32, device=device))[0]
                          .softmax(dim=0)[1])
                res[WSI[idx[id(v)]]] = (p, CLASSES[int(p > 0.5)])
        out[seed] = res
        print(f"[MIL-Adapter seed {seed}] trained, {len(res)} eval slides scored")
    return out


def load_per_slide(path):
    d = pd.read_csv(path)
    need = {"slide_id", "pred_label", "score"}
    if not need <= set(d.columns):
        raise SystemExit(f"{path}: needs columns {sorted(need)}")
    if "seed" in d:
        return {int(s): {w: (sc, pr) for w, sc, pr in zip(g.slide_id, g.score, g.pred_label)}
                for s, g in d.groupby("seed")}
    return {None: {w: (sc, pr) for w, sc, pr in zip(d.slide_id, d.score, d.pred_label)}}


# =====================================================================================
# Evaluation
# =====================================================================================

def evaluate(methods, meta, wit, seeds, k_shots, calibrate, calib_q):
    """methods: {label: (kind, {seed|None: {wsi: (score, pred)}})}, kind in
    {"trained", "zero_shot"}. Returns per-slide rows (seed x method x slide)."""
    Y = np.array([CLASSES.index(g) for g in meta.GT])
    wsis = meta.WSI.to_numpy()
    gt = dict(zip(meta.WSI, meta.GT))
    rows, notes = [], []
    for seed in seeds:
        tr, va = TRM.fewshot_split_ids(Y, k_shots, seed)
        eval_ids, support = set(wsis[va]), wsis[tr]
        per = {}
        for label, (kind, by_seed) in methods.items():
            sc = by_seed.get(seed, by_seed.get(None))
            if sc is None:
                raise SystemExit(f"{label}: no scores for seed {seed}")
            if None not in by_seed and set(sc) != eval_ids:
                raise SystemExit(f"{label}, seed {seed}: its slides differ from the k={k_shots} "
                                 "eval split -- different split or k.")
            thr = None
            if calibrate and kind == "zero_shot":
                neg = [sc[w][0] for w in support if gt[w] == "Normal" and w in sc]
                thr = float(np.quantile(neg, calib_q)) if neg else None
            per[label] = (sc, thr)
        common = set.intersection(eval_ids, *[set(s) for s, _ in per.values()])
        if len(common) < len(eval_ids):
            notes.append(f"seed {seed}: {len(eval_ids) - len(common)} eval slides lack a score "
                         "in some method; all methods use the common slides")
        for label, (sc, thr) in per.items():
            for w in common:
                s, pred = sc[w]
                if thr is not None:
                    pred = CLASSES[int(s > thr)]
                r = {"seed": seed, "method": label, "WSI": w, "gt_label": gt[w],
                     "score": float(s), "pred": pred, "correct": float(pred == gt[w])}
                if gt[w] == "Tumor" and w in wit.index:
                    r.update(wit.loc[w].to_dict())
                rows.append(r)
    return pd.DataFrame(rows), notes


def placement_auc(pos, neg):
    return TRM.auc_from_placement(TRM.placement_values(pos, neg)) if len(pos) and len(neg) else np.nan


def summarise(df, method_order):
    """Per seed and method: overall + per bin + per stage; then mean / SD over seeds."""
    seed_rows = []
    for (seed, m), g in df.groupby(["seed", "method"]):
        t, n = g[g.gt_label == "Tumor"], g[g.gt_label == "Normal"]
        sens, spec = t.correct.mean(), n.correct.mean()
        r = {"seed": seed, "method": m, "accuracy": g.correct.mean(),
             "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens, "specificity": spec,
             "auc": placement_auc(t.score.values, n.score.values)}
        for col, cats in (("wr_bin", WR_LABELS), ("lesion_stage", STAGES)):
            for c in cats:
                s = t[t[col] == c]
                r[f"det[{c}]"] = s.correct.mean() if len(s) else np.nan
                r[f"auc[{c}]"] = placement_auc(s.score.values, n.score.values)
                r[f"n[{c}]"] = len(s)
        seed_rows.append(r)
    ps = pd.DataFrame(seed_rows)

    def table(cats):
        out = []
        for m in method_order:
            g = ps[ps.method == m]
            for c in ["Normal"] + cats:
                det = g.specificity if c == "Normal" else g[f"det[{c}]"]
                auc = pd.Series(dtype=float) if c == "Normal" else g[f"auc[{c}]"]
                n = (df[(df.method == m) & (df.gt_label == "Normal")].groupby("seed").size()
                     if c == "Normal" else g[f"n[{c}]"])
                out.append({"method": m, "group": c, "n_slides_per_seed": float(np.mean(n)),
                            "detection_rate_mean": det.mean(), "detection_rate_sd": det.std(),
                            "auc_mean": auc.mean() if len(auc) else np.nan,
                            "auc_sd": auc.std() if len(auc) else np.nan})
        return pd.DataFrame(out)

    return ps, table(WR_LABELS), table(STAGES)


# =====================================================================================
# Figures
# =====================================================================================

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_groups(tab, groups, method_order, per_seed, xlabel, title, out_png, footer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    nm = len(method_order)
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8),
                               gridspec_kw={"width_ratios": [1.25, 1]})
    cats = ["Normal"] + groups
    x = np.arange(len(cats))
    width = 0.8 / nm
    n_lab = tab[tab.method == method_order[0]].set_index("group").n_slides_per_seed
    for i, m in enumerate(method_order):
        d = tab[tab.method == m].set_index("group").reindex(cats)
        xs = x + (i - (nm - 1) / 2) * width
        a.bar(xs, d.detection_rate_mean, width * 0.9,
              yerr=d.detection_rate_sd.fillna(0), color=PALETTE[i % 8],
              error_kw={"ecolor": INK2, "elinewidth": 0.9, "capsize": 2},
              label=m)
        for xv, yv in zip(xs, d.detection_rate_mean):      # a zero-height bar is invisible
            if np.isfinite(yv) and yv < 0.005:
                a.text(xv, 0.012, "0", ha="center", va="bottom", fontsize=7, color=PALETTE[i % 8])
    a.axvline(0.5, color=GRID, lw=1.5)
    a.set_xticks(x)
    a.set_xticklabels([f"{c}\n(n≈{n_lab.get(c, 0):.0f})" for c in cats])
    a.set_ylim(0, 1.05)
    a.set_ylabel("share of slides classified correctly")
    a.set_xlabel(xlabel)
    a.set_title("Accuracy: normal slides kept Normal | tumor slides detected", loc="left",
                fontsize=10, color=INK)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.set_axisbelow(True)

    xb = np.arange(len(groups))
    for i, m in enumerate(method_order):
        d = tab[tab.method == m].set_index("group").reindex(groups)
        y, e = d.auc_mean.to_numpy(float), d.auc_sd.fillna(0).to_numpy(float)
        ok = np.isfinite(y)
        b.fill_between(xb[ok], (y - e)[ok], np.minimum(1, (y + e)[ok]), color=PALETTE[i % 8],
                       alpha=0.12, lw=0)
        b.plot(xb[ok], y[ok], "o-", color=PALETTE[i % 8], lw=2.2, ms=6, mec="white", mew=0.8,
               label=m)
    b.axhline(0.5, color=INK2, lw=1, ls="--")
    b.text(len(groups) - 1, 0.51, "chance", ha="right", va="bottom", fontsize=7.5, color=INK2)
    b.set_xticks(xb)
    b.set_xticklabels(groups)
    b.set_xlim(-0.4, len(groups) - 0.6)
    b.set_ylim(0.3, 1.02)
    b.set_ylabel("AUC vs all normal slides")
    b.set_xlabel(xlabel)
    b.set_title("Ranking, threshold-free (AUC)", loc="left", fontsize=10, color=INK)
    b.grid(axis="y", color=GRID, lw=0.6)
    b.set_axisbelow(True)

    ov = per_seed.groupby("method")[["balanced_accuracy", "auc"]].mean()
    handles, _ = a.get_legend_handles_labels()
    fig.legend(handles, [f"{m}   (balanced acc {ov.loc[m, 'balanced_accuracy']:.3f}, "
                         f"AUC {ov.loc[m, 'auc']:.3f})" for m in method_order],
               loc="lower center", ncol=min(nm, 2), frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
    rows_leg = int(np.ceil(nm / 2))
    fig.text(0.01, 0.005 + 0.045 * rows_leg + 0.02, footer, fontsize=7.2, color=INK2)
    fig.tight_layout(rect=(0, 0.07 + 0.045 * rows_leg, 1, 0.95))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# =====================================================================================
# Main
# =====================================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv_path", default="./local_data/csv/CAMELYON16.csv")
    p.add_argument("--folder", help="MIL-format data root (needed to compute MI-Zero or train)")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--mizero_k", type=float, nargs="*", default=[],
                   help="Compute MI-Zero top-K from the features for these K.")
    p.add_argument("--train_miladapter", action="store_true")
    p.add_argument("--adapter", default="TaskRes",
                   choices=["ZSMIL", "TaskRes", "CLIPAdapter", "TIPAdapter"])
    p.add_argument("--init", choices=["ZS", "random"], default="ZS")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--per_slide", nargs="*", default=[],
                   help='Precomputed methods, "LABEL=path". A label containing "MI-Zero" '
                        'or "zero-shot" is treated as zero-shot for --calibrate_zero_shot.')
    p.add_argument("--k_shots", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--calibrate_zero_shot", action="store_true")
    p.add_argument("--calib_quantile", type=float, default=0.95)
    p.add_argument("--lesion_csv", default=None)
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--xml_dir", default=None)
    p.add_argument("--patch_size_level0", type=int, default=1024)
    p.add_argument("--mpp_level0", type=float, default=0.243)
    p.add_argument("--out_prefix", default="analysis/analysis_compare/compare")
    args = p.parse_args()

    meta = pd.read_csv(args.csv_path)

    # ---- witness rate and stage --------------------------------------------------------
    if {"tumor_area_pct", "lesion_stage"} <= set(meta.columns):
        wit = meta.set_index("WSI").loc[meta.GT.values == "Tumor",
                                        ["tumor_area_pct", "lesion_stage"]]
        wsrc = f"{args.csv_path} (tumor_area_pct, lesion_stage)"
    elif args.lesion_csv:
        wit = witness_from_lesion_csv(meta, args.lesion_csv, args.patch_size_level0,
                                      args.mpp_level0)
        wsrc = (f"lesion areas from {os.path.basename(args.lesion_csv)}, tissue = n_patches x "
                f"({args.patch_size_level0} px x {args.mpp_level0} um)^2")
    elif args.xml_dir:
        wit = witness_from_annotations(meta, args.h5_dir, args.xml_dir, args.mpp_level0)
        wsrc = "utils/witness_rate.py from h5 + XML"
    else:
        raise SystemExit("No witness-rate source: add tumor_area_pct/lesion_stage to the CSV "
                         "(add_tumor_pct_to_csv.py), or pass --lesion_csv or --xml_dir.")
    wit = wit.copy()
    wit["wr_bin"] = pd.cut(wit.tumor_area_pct, WR_BINS, labels=WR_LABELS,
                           include_lowest=True).astype(str)

    # ---- methods -----------------------------------------------------------------------
    methods = {}
    if args.mizero_k or args.train_miladapter:
        if not args.folder:
            raise SystemExit("--folder is needed to compute MI-Zero or train MIL-Adapter")
        from utils.utils import load_data
        X, Y, WSI = load_data(folder=args.folder, project=args.project, encoder=args.encoder,
                              classes=CLASSES)
        if list(WSI) != list(meta.WSI):
            raise SystemExit("load_data order differs from --csv_path")
        if args.train_miladapter:
            methods[f"MIL-Adapter ({args.adapter}+ABMIL)"] = (
                "trained", train_miladapter_scores(X, Y, WSI, args.seeds, args.k_shots, args))
        protos = np.load(args.text_prototypes)
        for k in args.mizero_k:
            res = mizero_scores(X, protos, k)
            methods[f"MI-Zero top-{k:g}" + ("" if k >= 1 else " of N")] = (
                "zero_shot", {None: dict(zip(WSI, res))})
    for spec in args.per_slide:
        label, path = spec.rsplit("=", 1)
        kind = "zero_shot" if any(s in label.lower() for s in ("mi-zero", "zero-shot", "zeroshot")) \
            else "trained"
        methods[label] = (kind, load_per_slide(path))
    if not methods:
        raise SystemExit("No method: use --mizero_k, --train_miladapter and/or --per_slide")
    order = list(methods)

    # ---- evaluate ------------------------------------------------------------------------
    df, notes = evaluate(methods, meta, wit, args.seeds, args.k_shots,
                         args.calibrate_zero_shot, args.calib_quantile)
    per_seed, by_bin, by_stage = summarise(df, order)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)) or ".", exist_ok=True)
    df.to_csv(f"{args.out_prefix}_per_slide.csv", index=False)
    per_seed.to_csv(f"{args.out_prefix}_per_seed.csv", index=False)
    by_bin.to_csv(f"{args.out_prefix}_by_bin.csv", index=False)
    by_stage.to_csv(f"{args.out_prefix}_by_stage.csv", index=False)

    n_t = df[(df.gt_label == "Tumor") & (df.method == order[0])].groupby("seed").size().mean()
    n_n = df[(df.gt_label == "Normal") & (df.method == order[0])].groupby("seed").size().mean()
    thr_note = (f"Zero-shot threshold: {args.calib_quantile:.0%} quantile of each seed's support "
                "normal slides." if args.calibrate_zero_shot else
                "Zero-shot labels: argmax of pooled similarities (uncalibrated).")
    footer = (f"{len(args.seeds)} few-shot splits (k={args.k_shots}); per split ≈{n_t:.0f} tumor / "
              f"{n_n:.0f} normal eval slides, support slides excluded for every method. Bars and "
              f"lines: mean over splits; error bars / bands: ± SD.\n{thr_note}  Witness rate: "
              f"{wsrc}.")
    tag = f"k={args.k_shots}, {len(args.seeds)} splits"
    plot_groups(by_bin, WR_LABELS, order, per_seed, "tumor area, % of tissue",
                f"Slide-level performance by witness rate ({tag})",
                f"{args.out_prefix}_by_witness_rate.png", footer)
    plot_groups(by_stage, STAGES, order, per_seed, "largest lesion (ITC < 0.2 mm ≤ micro < 2 mm ≤ macro)",
                f"Slide-level performance by lesion stage ({tag})",
                f"{args.out_prefix}_by_stage.png", footer)

    # ---- report --------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"Slide-level comparison  [{tag}; ≈{n_t:.0f} tumor / {n_n:.0f} normal eval slides per split]")
    print("=" * 100)
    ov = per_seed.groupby("method")[["accuracy", "balanced_accuracy", "sensitivity",
                                     "specificity", "auc"]].agg(["mean", "std"]).reindex(order)
    print(ov.to_string(float_format=lambda v: f"{v:.3f}"))
    for name, tab in (("witness rate (tumor area)", by_bin), ("lesion stage", by_stage)):
        print("-" * 100 + f"\nby {name}: detection rate (Normal = specificity) | AUC, mean over splits")
        piv = tab.pivot(index="group", columns="method", values="detection_rate_mean") \
                 .reindex(["Normal"] + (WR_LABELS if tab is by_bin else STAGES))[order]
        pa = tab.pivot(index="group", columns="method", values="auc_mean") \
                .reindex(piv.index)[order]
        piv.columns = [f"det | {c}" for c in piv.columns]
        pa.columns = [f"AUC | {c}" for c in pa.columns]
        print(pd.concat([piv, pa], axis=1).to_string(float_format=lambda v: f"{v:.3f}"))
    for n in notes:
        print("NOTE:", n)
    print("=" * 100)
    print(f"-> {args.out_prefix}_by_witness_rate.png, _by_stage.png, _per_slide.csv, _per_seed.csv, "
          f"_by_bin.csv, _by_stage.csv")


if __name__ == "__main__":
    main()
