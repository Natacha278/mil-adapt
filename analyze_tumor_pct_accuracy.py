"""
analyze_tumor_pct_accuracy.py

For one seed, trains a MIL-Adapter few-shot model on CAMELYON16 and measures how its
slide-level accuracy varies with the PERCENTAGE OF TUMOR PATCHES on each evaluated slide
(i.e. how focal/small the tumor region is, as a fraction of tissue patches).

Place this file next to main.py (same as visu_ABMIL.py) -- it imports shared pieces
(ADAPTER_CLASSES, patch_known_bugs, get_project_data_camelyon16) directly from
visu_ABMIL.py rather than duplicating them.

TUMOR-PATCH PERCENTAGE IS READ, NOT RECOMPUTED: this script expects the CSV
(local_data/csv/CAMELYON16.csv by default) to already have a tumor_patch_pct column,
added by add_tumor_pct_to_csv.py. Run that script once first. This avoids re-scanning
every h5 + XML file on every analysis run -- the percentage doesn't change between runs,
only the trained model/seed does.

By default (--tumor_only, on), the accuracy-vs-tumor-pct% analysis is restricted to
GT Tumor slides only: Normal slides all sit at 0% tumor-patch by construction, so
lumping them into the same bin as small-lesion Tumor slides would conflate
"accuracy on Normal slides" with "accuracy as the tumor region shrinks". Normal-slide
accuracy is still reported separately in the console output.

BINNING: default is QUANTILE binning (--binning quantile, --n_bins), which gives each
bin roughly the same number of slides -- fixed-width bins (the old default) are usually
very unevenly populated since tumor-patch-% isn't uniformly distributed across slides,
which makes per-bin accuracy hard to compare/trust. Slides at exactly 0% tumor-patch
(should only be a handful of Tumor slides whose lesion didn't register at any patch
center -- see add_tumor_pct_to_csv.py's own sanity check) get their own "0%" bin,
kept separate from the quantile split of the nonzero slides. --binning fixed reverts
to explicit --bin_edges.

Run on a GPU allocation (salloc/sbatch), not the login node -- this trains the model.
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils import set_random_seeds, load_data, fewshot_sampling
from utils.trainer import train_model
from visu_ABMIL_zs import (
    ADAPTER_CLASSES,
    UNIMPLEMENTED_ADAPTERS,
    patch_known_bugs,
    get_project_data_camelyon16,
)


# =====================================================================================
# Binning
# =====================================================================================

def assign_quantile_bins(df, pct_col, n_bins, zero_own_bin, precision=2):
    """
    Assigns each row a tumor_pct_bin label using quantile edges (roughly equal slide
    count per bin), computed over the NONZERO values only when zero_own_bin=True so a
    handful of exact-0% slides don't distort/collapse the low end of the quantile split.
    Returns (df_with_bin_col, ordered_list_of_bin_labels).
    """
    df = df.copy()
    zero_mask = df[pct_col] == 0 if zero_own_bin else pd.Series(False, index=df.index)
    nonzero_df = df.loc[~zero_mask]

    bin_order = []
    df["tumor_pct_bin"] = None

    if zero_mask.any():
        df.loc[zero_mask, "tumor_pct_bin"] = "0%"
        bin_order.append("0%")

    if len(nonzero_df) > 0:
        n_bins_eff = min(n_bins, nonzero_df[pct_col].nunique())
        if n_bins_eff < 1:
            n_bins_eff = 1
        qbinned = pd.qcut(nonzero_df[pct_col], q=n_bins_eff, duplicates="drop", precision=precision)
        labels = qbinned.astype(str)
        df.loc[nonzero_df.index, "tumor_pct_bin"] = labels

        # Interval categories from qcut sort correctly on their own; recover that order
        # for the (now stringified) labels via each interval's left edge.
        unique_intervals = sorted(qbinned.cat.categories, key=lambda iv: iv.left)
        bin_order.extend(str(iv) for iv in unique_intervals)

    return df, bin_order


def assign_fixed_bins(df, pct_col, bin_edges):
    """Original fixed-width binning, kept as an explicit opt-in via --binning fixed."""
    df = df.copy()
    bin_labels = [f"[{bin_edges[i]:g},{bin_edges[i+1]:g})" for i in range(len(bin_edges) - 2)]
    bin_labels.append(f"[{bin_edges[-2]:g},{bin_edges[-1]:g}]")

    df["tumor_pct_bin"] = pd.cut(
        df[pct_col], bins=bin_edges, labels=bin_labels, include_lowest=True, right=False
    )
    df.loc[df[pct_col] == bin_edges[-1], "tumor_pct_bin"] = bin_labels[-1]
    return df, bin_labels


# =====================================================================================
# Main
# =====================================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default=None,
                         help="CAMELYON16 CSV with a precomputed tumor_patch_pct column "
                              "(run add_tumor_pct_to_csv.py first if it's missing). "
                              "Defaults to ./local_data/csv/<project>.csv, matching the "
                              "path load_data() itself reads.")
    parser.add_argument("--folder", required=True,
                         help="MIL-Adapter data root (--folder in main.py): folder/project/encoder/*.npy")
    parser.add_argument("--project", default="CAMELYON16")
    parser.add_argument("--encoder", default="CONCH")
    parser.add_argument("--adapter", default="ZSMIL",
                         choices=["ZSMIL", "TaskRes", "CLIPAdapter", "TIPAdapter"])
    parser.add_argument("--aggregator", default="ABMIL",
                         choices=["BGAP", "BGMP", "ABMIL", "TransMIL", "WIKGMIL", "ILRAMIL", "RRTMIL"])
    parser.add_argument("--init", choices=["ZS", "random"], default="random",
                         help="Only used by --adapter ZSMIL.")
    parser.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    parser.add_argument("--k_shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="The single seed to analyze.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_set", choices=["val", "train", "all"], default="val",
                         help="Which slides to evaluate accuracy on. 'val' (default) is the "
                              "proper held-out few-shot evaluation set for this seed; 'train' "
                              "includes the k-shot support slides the model was fit on "
                              "(inflated/optimistic accuracy); 'all' is both.")
    parser.add_argument("--binning", choices=["quantile", "fixed"], default="quantile",
                         help="'quantile' (default): bin edges chosen so each bin has "
                              "roughly the same number of slides -- fixes the very unequal "
                              "bin sizes fixed-width edges produce. 'fixed': use --bin_edges "
                              "explicitly instead.")
    parser.add_argument("--n_bins", type=int, default=5,
                         help="Number of quantile bins for the NONZERO tumor-patch-%% slides "
                              "(only used with --binning quantile). The 0%% slides, if any, "
                              "always get their own separate bin -- see --zero_own_bin.")
    parser.add_argument("--bin_edges", type=float, nargs="+",
                         default=[0, 1, 5, 10, 25, 50, 100],
                         help="Bin edges (percent), only used with --binning fixed.")
    parser.add_argument("--zero_own_bin", dest="zero_own_bin", action="store_true", default=True,
                         help="(quantile binning only) Give exact-0%% slides their own bin "
                              "instead of folding them into the lowest quantile bin. Default: on.")
    parser.add_argument("--no_zero_own_bin", dest="zero_own_bin", action="store_false")
    parser.add_argument("--tumor_only", dest="tumor_only", action="store_true", default=True,
                         help="Restrict the accuracy-vs-tumor-pct%% analysis to GT Tumor slides "
                              "only (default: on). Normal slides all sit at 0%% tumor-patch by "
                              "construction, so lumping them into the same bin as small-lesion "
                              "Tumor slides would conflate 'accuracy on Normal slides' with "
                              "'accuracy as the tumor region shrinks'.")
    parser.add_argument("--no_tumor_only", dest="tumor_only", action="store_false",
                         help="Include Normal slides too (reverts to the old, conflated "
                              "0%% bin). Normal-slide accuracy is reported separately either way.")
    parser.add_argument("--out_dir", default="./analysis")
    parser.add_argument("--out_prefix", default=None,
                         help="Filename prefix for outputs. Defaults to "
                              "'<project>_<adapter>_<aggregator>_seed<seed>'.")
    args = parser.parse_args()

    if args.adapter in UNIMPLEMENTED_ADAPTERS:
        raise ValueError(f"--adapter {args.adapter} is not implemented in MIL-Adapter's utils/adapters.py.")

    os.makedirs(args.out_dir, exist_ok=True)
    out_prefix = args.out_prefix or f"{args.project}_{args.adapter}_{args.aggregator}_seed{args.seed}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU visible — utils/trainer.py hardcodes .cuda(), this will fail on CPU.")

    # ---------------------------------------------------------------------------
    # 0) Load precomputed tumor-patch percentages straight from the CSV
    # ---------------------------------------------------------------------------
    csv_path = args.csv_path or f"./local_data/csv/{args.project}.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    csv_df = pd.read_csv(csv_path)
    if "tumor_patch_pct" not in csv_df.columns:
        raise ValueError(
            f"'{csv_path}' has no tumor_patch_pct column. Run add_tumor_pct_to_csv.py first "
            "to precompute it (this script reads it rather than recomputing it every run)."
        )
    tumor_pct_by_slide = dict(zip(csv_df["WSI"], csv_df["tumor_patch_pct"]))

    # ---------------------------------------------------------------------------
    # 1) Load data, run the SAME few-shot split/training as main.py for this seed
    # ---------------------------------------------------------------------------
    classes, classes_id = get_project_data_camelyon16()
    text_prototypes_np = np.load(args.text_prototypes)

    X, Y, WSI = load_data(folder=args.folder, project=args.project,
                           encoder=args.encoder, classes=classes)

    missing = [s for s in WSI if s not in tumor_pct_by_slide]
    if missing:
        raise ValueError(
            f"{len(missing)} slide(s) in {csv_path} have no tumor_patch_pct (e.g. {missing[:5]}). "
            "Re-run add_tumor_pct_to_csv.py against the current CSV."
        )

    set_random_seeds(seed_value=args.seed)
    train_data, val_data, train_labels, val_labels = fewshot_sampling(
        X=X, Y=Y, k_shots=args.k_shots, seed=args.seed
    )

    # Recover original slide indices for train/val splits via object identity
    # (fewshot_sampling returns "X[i] for i in ids", i.e. the SAME array objects,
    # so this is exact -- no need to duplicate fewshot_sampling's sampling logic).
    id_to_idx = {id(x): i for i, x in enumerate(X)}
    train_ids = np.array([id_to_idx[id(v)] for v in train_data])
    val_ids = np.array([id_to_idx[id(v)] for v in val_data])

    adapter_cls = ADAPTER_CLASSES[args.adapter]
    if args.adapter == "ZSMIL":
        model = adapter_cls(text_embeddings=text_prototypes_np, aggregator=args.aggregator, init=args.init)
    elif args.adapter in ("TaskRes", "CLIPAdapter"):
        model = adapter_cls(text_embeddings=text_prototypes_np, aggregator=args.aggregator)
    elif args.adapter == "TIPAdapter":
        model = adapter_cls(text_embeddings=text_prototypes_np, train_data=train_data,
                             train_labels=train_labels, aggregator=args.aggregator)
    model.to(device)
    model = patch_known_bugs(model, args.adapter, args.aggregator)

    weight_decay, adamw_beta, epochs = 1e-5, (0.9, 0.999), args.epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=adamw_beta, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    train_model(model, optimizer, criterion, scheduler, train_data, train_labels, epochs)

    # ---------------------------------------------------------------------------
    # 2) Choose which slides to evaluate, and get the model's prediction for each
    # ---------------------------------------------------------------------------
    if args.eval_set == "val":
        eval_ids = val_ids
    elif args.eval_set == "train":
        eval_ids = train_ids
    else:
        eval_ids = np.concatenate([train_ids, val_ids])

    model.eval()
    predictions = []
    with torch.no_grad():
        for idx in tqdm(eval_ids, desc=f"Predicting on {args.eval_set} slides"):
            batch = torch.tensor(X[idx], dtype=torch.float32, device=device)
            logits = model(batch)[0]
            pred_id = int(logits.softmax(dim=0).argmax().item())
            predictions.append(pred_id)
    predictions = np.array(predictions)

    # ---------------------------------------------------------------------------
    # 3) Assemble per-slide results table (tumor-patch %% read from CSV, not recomputed)
    # ---------------------------------------------------------------------------
    rows = []
    for i, idx in enumerate(eval_ids):
        slide_id = WSI[idx]
        gt_id = int(Y[idx])
        pred_id = int(predictions[i])
        rows.append({
            "slide_id": slide_id,
            "split": "train" if idx in train_ids else "val",
            "gt_label": classes_id[gt_id],
            "pred_label": classes_id[pred_id],
            "correct": gt_id == pred_id,
            "tumor_patch_pct": tumor_pct_by_slide[slide_id],
        })
    df_all = pd.DataFrame(rows)  # both classes, kept for the saved CSV and the Normal-slide summary

    csv_out_path = os.path.join(args.out_dir, f"{out_prefix}_per_slide.csv")
    df_all.to_csv(csv_out_path, index=False)
    print(f"Saved per-slide results (Normal + Tumor): {csv_out_path}")

    df_normal = df_all[df_all["gt_label"] == "Normal"]
    if len(df_normal) > 0:
        print(f"Normal-slide accuracy (n={len(df_normal)}): {df_normal['correct'].mean():.4f}")
    else:
        print("No Normal slides in this eval_set.")

    # ---------------------------------------------------------------------------
    # 4) Bin by tumor-patch percentage and compute accuracy per bin
    # ---------------------------------------------------------------------------
    df = df_all[df_all["gt_label"] == "Tumor"].copy() if args.tumor_only else df_all.copy()
    if args.tumor_only:
        print(f"\nRestricting tumor-pct-vs-accuracy analysis to Tumor slides only (n={len(df)}).")

    if args.binning == "quantile":
        df, bin_labels = assign_quantile_bins(df, "tumor_patch_pct", args.n_bins, args.zero_own_bin)
    else:
        df, bin_labels = assign_fixed_bins(df, "tumor_patch_pct", args.bin_edges)

    bin_stats = df.groupby("tumor_pct_bin", observed=True).agg(
        n_slides=("correct", "size"),
        n_correct=("correct", "sum"),
        n_tumor_gt=("gt_label", lambda s: (s == "Tumor").sum()),
        n_normal_gt=("gt_label", lambda s: (s == "Normal").sum()),
    )
    bin_stats["accuracy"] = bin_stats["n_correct"] / bin_stats["n_slides"]
    bin_stats = bin_stats.reindex(bin_labels)

    stats_csv_path = os.path.join(args.out_dir, f"{out_prefix}_bin_accuracy.csv")
    bin_stats.to_csv(stats_csv_path)
    print(f"Saved per-bin accuracy: {stats_csv_path}")
    print(f"\nAccuracy by tumor-patch percentage bin ({args.binning} binning):")
    print(bin_stats.to_string())

    overall_acc = df["correct"].mean()
    print(f"\nOverall accuracy on {args.eval_set} set (n={len(df)}): {overall_acc:.4f}")

    # ---------------------------------------------------------------------------
    # 5) Plot: accuracy per bin, with sample count annotated on each bar
    # ---------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(bin_labels))
    accs = bin_stats["accuracy"].values
    ns = bin_stats["n_slides"].values

    ax.bar(x, np.nan_to_num(accs), color="#3b6fa0", edgecolor="black", width=0.65)
    ax.axhline(overall_acc, color="darkred", linestyle="--", linewidth=1.2,
               label=f"Overall accuracy ({overall_acc:.2f})")

    for xi, acc, n in zip(x, accs, ns):
        if np.isnan(acc):
            continue
        ax.text(xi, acc + 0.02, f"n={int(n)}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right")
    ax.set_xlabel("Tumor-patch percentage on slide (%)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.15)
    ax.set_title(f"{args.adapter}+{args.aggregator} accuracy vs. tumor-patch %"
                 f"  ({args.project}, seed={args.seed}, k_shots={args.k_shots}, {args.eval_set} set)")
    ax.legend(loc="lower right")
    plt.tight_layout()

    plot_path = os.path.join(args.out_dir, f"{out_prefix}_accuracy_vs_tumor_pct.png")
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
