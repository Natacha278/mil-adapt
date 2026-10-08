"""
analyze_tumor_ratio_mizero.py

MI-Zero (Lu et al., CVPR 2023) performance as a function of tumor ratio: per-patch CONCH
image-text similarities pooled with a TOP-K MEAN per class, then the same
discrimination-vs-burden analysis the MIL-Adapter arm uses.

Self-contained: scores the slides from the TRIDENT .h5 features and writes its own
per-slide CSV. Nothing precomputed, no main.py, no trained model -- MI-Zero is
training-free.

WHY THIS BASELINE
-----------------
ABMIL and TransMIL already both show the low-burden drop, so a third attention-based
aggregator is a replication rather than a test of mechanism. Every aggregator in
utils/MIL/ builds the bag representation as a convex combination of instances -- a
mean-like functional -- and a mean-like statistic has asymptotically zero power once the
contaminated fraction pi falls below N^(-1/2). MI-Zero is the cheap baseline in a
DIFFERENT pooling class: an order statistic, near-optimal exactly where the mean fails
and suboptimal where the mean is fine. Its predicted curve therefore differs
QUALITATIVELY, which makes it discriminating rather than confirmatory:

  * flatter at low burden, worse at high burden -> the pooling regime is the mechanism;
  * drops just as much at every K -> the cause is NOT pooling. Suspect the encoder or
    the prompt at patch level, and analyze_zeroshot_score_vs_lesion_size.py decides it;
  * scores correlated with slide size among normals -> extreme-value drift of a max-type
    statistic (E[max of N nulls] grows like sqrt(2 log N)). Reported per K as
    `spearman_score_vs_npatches_normal`.

NOT the same as --adapter ZSMIL --aggregator BGMP. That path does
`torch.max(features, dim=0)`, an element-wise max over EMBEDDING DIMENSIONS, producing a
bag vector no patch ever had. MI-Zero takes an order statistic over per-patch CLASS
SCORES. Different object, different statistical behaviour.

K IS THE ASSUMED SPARSITY, so it is swept. --topk takes integers and fractions (a value
< 1 is a fraction of that slide's patch count, so K scales with slide size). All values
are evaluated in ONE pass over the features.

MATCHING THE MIL-ADAPTER EVAL SET
---------------------------------
MI-Zero has no support set, so by default it scores every slide. Pass --k_shots and
--seeds to exclude each seed's support slides using the same split
(utils.tumor_ratio_metrics.fewshot_split_ids, which analyze_tumor_ratio_miladapter.py
asserts against the real fewshot_sampling on live data). That reproduces the split from
the labels alone, without loading the .npy bag features MI-Zero never needs.

OUTPUT (per K)
--------------
  <prefix>_k<K>_per_slide.csv / _bin_auc.csv / _summary.json / _auc_vs_tumor_pct.png
plus <prefix>_sweep.csv and <prefix>_sweep.png comparing the K values on shared axes.

USAGE
-----
    python analyze_tumor_ratio_mizero.py --self_test
    python analyze_tumor_ratio_mizero.py \
        --topk 1 5 20 50 100 0.01 \
        --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out_dir analysis_mizero

Needs a GPU allocation for the CONCH projection. --self_test does not.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import tumor_ratio_metrics as M

CLASSES = ["Normal", "Tumor"]          # matches get_project_data_camelyon16()
NORMAL_IDX, TUMOR_IDX = 0, 1
CONCH_LOGIT_SCALE = 56.347694396972656  # same constant as utils/adapters.py ZSMIL


# =====================================================================================
# Pooling (numpy, so --self_test needs neither torch nor CONCH)
# =====================================================================================

def resolve_k(k, n_patches):
    """K as an int, or as a fraction of this slide's patch count when 0 < k < 1."""
    kk = int(np.ceil(k * n_patches)) if 0 < k < 1 else int(k)
    return int(np.clip(kk, 1, n_patches))


def topk_pool(sim, k, mode="logit"):
    """MI-Zero slide score per class: mean of that class's top-K patch scores.

    mode 'logit' pools the similarities directly (MI-Zero as published); 'prob' applies
    a per-patch softmax over classes first (scaled by CONCH's logit_scale) and pools
    probabilities. Pooling probabilities couples the classes through the softmax, so the
    two are NOT monotone transforms of each other -- hence both are exposed rather than
    assumed equivalent.
    """
    sim = np.asarray(sim, dtype=np.float64)
    n = sim.shape[0]
    kk = resolve_k(k, n)
    x = sim
    if mode == "prob":
        z = sim * CONCH_LOGIT_SCALE
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        x = e / e.sum(axis=1, keepdims=True)
    part = np.partition(x, n - kk, axis=0)[n - kk:]
    return part.mean(axis=0), kk


def slide_score(sim, k, mode="logit"):
    """(margin, pred_id, k_used). margin = pooled Tumor - pooled Normal."""
    pooled, kk = topk_pool(sim, k, mode)
    return float(pooled[TUMOR_IDX] - pooled[NORMAL_IDX]), int(np.argmax(pooled)), kk


# =====================================================================================
# CONCH projection (lazy imports)
# =====================================================================================

def load_projector(checkpoint_path, text_prototypes_path, batch_size=4096):
    """sim_fn(features[N,D]) -> (N, C) cosine similarities.

    Identical pipeline to analyze_zeroshot_patch_accuracy.py: forward_project, L2
    normalize, dot with the already-normalized text prototypes.
    """
    import torch
    from conch.open_clip_custom import create_model_from_pretrained

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proto = np.load(text_prototypes_path)
    if proto.shape[0] < proto.shape[1]:          # stored [C, D] -> want [D, C]
        proto = proto.T
    if proto.shape[1] != len(CLASSES):
        raise SystemExit(f"{text_prototypes_path}: {proto.shape[1]} prototypes, "
                         f"expected {len(CLASSES)} for {CLASSES}")
    proto_t = torch.tensor(proto, dtype=torch.float32, device=device)

    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
    model = model.to(device).eval()

    @torch.no_grad()
    def sim_fn(features):
        feats = torch.as_tensor(features, dtype=torch.float32, device=device)
        out = []
        for i in range(0, feats.shape[0], batch_size):
            pz = model.visual.forward_project(feats[i:i + batch_size])
            out.append(torch.nn.functional.normalize(pz, dim=-1))
        return (torch.cat(out, 0) @ proto_t).cpu().numpy()

    return sim_fn


# =====================================================================================
# Self-test
# =====================================================================================

def self_test():
    M.self_test()
    print("self_test: analyze_tumor_ratio_mizero")
    rng = np.random.default_rng(0)

    sim = rng.normal(0, 1, (500, 2))
    assert np.allclose(topk_pool(sim, 1)[0], sim.max(0))
    assert np.allclose(topk_pool(sim, 500)[0], sim.mean(0))
    prev = np.inf
    for k in [1, 5, 20, 100, 500]:
        v = topk_pool(sim, k)[0][TUMOR_IDX]
        assert v <= prev + 1e-12
        prev = v
    assert resolve_k(0.01, 3775) == 38 and resolve_k(0.01, 900) == 9
    assert resolve_k(5, 3) == 3 and resolve_k(0.5, 1) == 1
    print("  K=1 == max, K=N == mean, monotone in K; fractional K scales with bag size")

    base = rng.normal(0, 1, (3000, 2)) * 0.1
    pos = base.copy(); pos[:6, TUMOR_IDX] += 3.0
    m_small = slide_score(pos, 5)[0] - slide_score(base, 5)[0]
    m_large = slide_score(pos, 1000)[0] - slide_score(base, 1000)[0]
    assert m_small > 10 * m_large
    dense = base.copy(); dense[:900, TUMOR_IDX] += 0.35
    d_small = slide_score(dense, 5)[0] - slide_score(base, 5)[0]
    d_large = slide_score(dense, 1000)[0] - slide_score(base, 1000)[0]
    assert d_large / max(d_small, 1e-9) > m_large / max(m_small, 1e-9)
    print(f"  sparse bag (6/3000): gain {m_small:.3f} at K=5 vs {m_large:.4f} at K=1000;"
          f"\n  dense bag (900/3000): gain {d_small:.3f} vs {d_large:.3f}"
          f"  <- no single K is best for both")

    sizes = [500, 1000, 2000, 4000, 8000]
    k1 = [np.mean([slide_score(rng.normal(0, 1, (n, 2)), 1)[0] for _ in range(60)]) for n in sizes]
    kf = [np.mean([slide_score(rng.normal(0, 1, (n, 2)), 0.1)[0] for _ in range(60)]) for n in sizes]
    assert np.std(k1) > np.std(kf)
    print(f"  null-bag drift with N: SD {np.std(k1):.3f} at K=1 vs {np.std(kf):.3f} at K=10%N")
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
    p.add_argument("--csv_path", default=None,
                   help="Project CSV (case_id, WSI, GT, tumor_patch_pct, n_patches). "
                        "Defaults to ./local_data/csv/<project>.csv.")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--checkpoint_path",
                   default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--topk", type=float, nargs="+", default=[1, 5, 20, 50, 100, 0.01])
    p.add_argument("--pool_mode", choices=["logit", "prob"], default="logit")
    p.add_argument("--k_shots", type=int, default=None, choices=[2, 4, 8, 16],
                   help="If given with --seeds, exclude each seed's few-shot support "
                        "slides so the eval set matches the MIL-Adapter arm exactly.")
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--bin_edges", type=float, nargs="+", default=M.DEFAULT_EDGES)
    p.add_argument("--wall_pct", type=float, default=None)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out_dir", default="./analysis_mizero")
    p.add_argument("--out_prefix", default=None)
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return

    import h5py
    from tqdm import tqdm
    from scipy.stats import spearmanr

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.out_prefix or f"{args.project}_MIZERO_{args.pool_mode}"
    csv_path = args.csv_path or f"./local_data/csv/{args.project}.csv"

    # Read the project CSV in the SAME order load_data() does, so slide indices line up
    # with the few-shot split without loading any .npy features.
    meta = pd.read_csv(csv_path)
    if "tumor_patch_pct" not in meta.columns:
        raise SystemExit(f"{csv_path} has no tumor_patch_pct. Run add_tumor_pct_to_csv.py first.")
    Y = np.array([CLASSES.index(g) for g in meta["GT"].values], dtype=np.int64)
    wall = (args.wall_pct if args.wall_pct is not None
            else M.detection_wall_pct(meta["n_patches"]) if "n_patches" in meta
            else np.nan)

    if args.limit:
        meta, Y = meta.iloc[:args.limit], Y[:args.limit]

    sim_fn = load_projector(args.checkpoint_path, args.text_prototypes, args.batch_size)

    rows, n_missing = [], 0
    for i, row in tqdm(list(meta.reset_index(drop=True).iterrows()), desc="MI-Zero"):
        h5_path = os.path.join(args.h5_dir, f"{row['WSI']}.h5")
        if not os.path.exists(h5_path):
            n_missing += 1
            continue
        with h5py.File(h5_path, "r") as f:
            sim = sim_fn(f["features"][:])
        for k in args.topk:
            margin, pred_id, kk = slide_score(sim, k, args.pool_mode)
            rows.append({"idx": i, "slide_id": row["WSI"], "gt_label": row["GT"],
                         "pred_label": CLASSES[pred_id], "score": margin,
                         "tumor_patch_pct": float(row["tumor_patch_pct"]),
                         "n_patches": sim.shape[0], "topk": k, "k_used": kk})
    if not rows:
        raise SystemExit("no slides scored -- check --h5_dir / --csv_path")
    df = pd.DataFrame(rows)

    # Eval splits: one per seed if asked to match the MIL-Adapter arm, else everything.
    if args.k_shots and args.seeds:
        splits = [(f"seed{s}", set(M.fewshot_split_ids(Y, args.k_shots, s)[1]))
                  for s in args.seeds]
    else:
        splits = [("all", set(df["idx"].unique()))]

    curves, sweep = [], []
    for k, gk in df.groupby("topk", sort=True):
        tag = f"{prefix}_k{k:g}"
        runs = []
        for name, keep in splits:
            g = gk[gk["idx"].isin(keep)].sort_values("idx")
            t, n = g[g.gt_label == "Tumor"], g[g.gt_label == "Normal"]
            runs.append((name, t["score"].to_numpy(),
                         t["tumor_patch_pct"].to_numpy(float), n["score"].to_numpy()))
        # MI-Zero is deterministic, so every split gives the same scores for a shared
        # slide; the bootstrap needs one slide set, so use the common intersection.
        common = set.intersection(*[keep for _, keep in splits])
        runs_common = []
        for name, _ in splits:
            g = gk[gk["idx"].isin(common)].sort_values("idx")
            t, n = g[g.gt_label == "Tumor"], g[g.gt_label == "Normal"]
            runs_common.append((name, t["score"].to_numpy(),
                                t["tumor_patch_pct"].to_numpy(float), n["score"].to_numpy()))

        bins_df, point, seed_sd, boot, summary = M.summarise(
            runs_common, args.bin_edges, wall, args.n_boot)
        summary.update(topk=k, median_k_used=float(gk["k_used"].median()),
                       pool_mode=args.pool_mode, text_prototypes=args.text_prototypes,
                       n_slides_evaluated=len(common))

        norm = gk[gk.gt_label == "Normal"]
        rho = float(spearmanr(norm["score"], norm["n_patches"]).statistic) if len(norm) > 2 else np.nan
        summary["spearman_score_vs_npatches_normal"] = rho

        gk.drop(columns=["topk", "idx"]).to_csv(
            os.path.join(args.out_dir, f"{tag}_per_slide.csv"), index=False)
        bins_df.to_csv(os.path.join(args.out_dir, f"{tag}_bin_auc.csv"), index=False)
        with open(os.path.join(args.out_dir, f"{tag}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        M.make_figure(bins_df.iloc[:-1], runs_common, args.bin_edges, wall,
                      os.path.join(args.out_dir, f"{tag}_auc_vs_tumor_pct.png"),
                      subtitle=f"MI-Zero top-K, K={k:g} ({args.pool_mode}); "
                               f"band = 95% paired bootstrap over slides")

        curves.append((f"K={k:g}", bins_df.iloc[:-1]))
        sweep.append({"topk": k, "median_k_used": summary["median_k_used"],
                      "auc_overall": summary["auc_overall"]["value"],
                      "auc_low": summary["auc_low"]["value"],
                      "auc_high": summary["auc_high"]["value"],
                      "auc_gap": summary["auc_gap"]["value"],
                      "slope_u_per_decade": summary["slope_u_per_decade"]["value"],
                      "spearman_score_vs_npatches_normal": rho,
                      "n_slides": summary["n_tumor"] + summary["n_normal"]})

    sweep_df = pd.DataFrame(sweep)
    sweep_path = os.path.join(args.out_dir, f"{prefix}_sweep.csv")
    sweep_df.to_csv(sweep_path, index=False)
    M.make_sweep_figure(curves, wall, os.path.join(args.out_dir, f"{prefix}_sweep.png"),
                        legend_title="top-$K$",
                        subtitle=f"MI-Zero, {args.pool_mode} pooling, "
                                 f"{sweep_df['n_slides'].iloc[0]} slides")

    print("\n" + "=" * 92)
    print(f"MI-Zero top-K vs tumor burden  [{args.project}, {args.pool_mode}, "
          f"{df.slide_id.nunique()} slides, {n_missing} missing h5]")
    print("=" * 92)
    print(sweep_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 92)
    print(f"per-K outputs -> {args.out_dir}/{prefix}_k*_*\nsweep -> {sweep_path}")
    print("\nauc_gap is AUC_high - AUC_low across the beta=1/2 wall: smaller means the\n"
          "statistic holds up better at low burden. If the gap shrinks as K falls while\n"
          "auc_high also falls, that is the order-statistic signature and the pooling\n"
          "account survives. If auc_gap is flat in K, pooling is not the mechanism.\n"
          "spearman_score_vs_npatches_normal clearly positive at small K means the\n"
          "statistic is partly reading slide size, not tumor.")


if __name__ == "__main__":
    main()
