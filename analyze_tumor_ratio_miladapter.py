"""
analyze_tumor_ratio_miladapter.py

MIL-Adapter performance as a function of tumor ratio, measured by DISCRIMINATION rather
than thresholded accuracy.

Self-contained: it runs the same few-shot split, model construction and training loop as
main.py, over as many seeds as you ask for, and keeps the continuous slide score. It
does NOT read a precomputed per-slide CSV and requires no change to main.py or
utils/trainer.py.

WHY IT HAS TO SCORE THE SLIDES ITSELF
-------------------------------------
validate_model() computes `test_logits.softmax(axis=0).argmax(axis=0)` and keeps only
the argmax, so the score is discarded on the line that produces it. That loss is not
recoverable, and it is fatal here rather than merely inconvenient: AUC computed from
binary predictions equals (TPR - FPR + 1)/2 = balanced accuracy exactly, and per
stratum equals (sens_bin + global specificity)/2 -- an affine rescaling of the
sensitivity curve analyze_tumor_pct_accuracy.py already produces. Reading pred_label
would give a differently-named copy of the old number, so the training loop is
reproduced here and the softmax probability of Tumor is retained.

WHAT IT PRODUCES
----------------
  <prefix>_per_slide.csv   every eval slide x seed, WITH the continuous score
  <prefix>_bin_auc.csv     per-burden-stratum AUC, bootstrap CI, seed SD, sens@spec
  <prefix>_summary.json    auc_overall / auc_low / auc_high / auc_gap / slope, with CIs
  <prefix>_auc_vs_tumor_pct.png

Metrics live in utils/tumor_ratio_metrics.py, shared verbatim with
analyze_tumor_ratio_mizero.py so the two arms are strictly comparable.

COST
----
Training is on precomputed .npy bag features, so a seed is minutes at k<=16 / 20
epochs; the expensive TRIDENT+CONCH feature extraction is untouched. load_data() reads
every slide's .npy once, up front, and that dominates wall time.

USAGE
-----
    python analyze_tumor_ratio_miladapter.py --self_test
    python analyze_tumor_ratio_miladapter.py \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
        --adapter TaskRes --aggregator ABMIL --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out_dir analysis_auc

Needs a GPU allocation. --self_test does not.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import tumor_ratio_metrics as M


def get_project_data_camelyon16():
    return ["Normal", "Tumor"], ["Normal", "Tumor"]


def build_model(adapter, text_prototypes, aggregator, init, train_data, train_labels):
    """Same construction main.py uses, including TIPAdapter's extra arguments."""
    from utils.adapters import ZSMIL, TaskRes, CLIPAdapter, TIPAdapter
    if adapter == "ZSMIL":
        return ZSMIL(text_embeddings=text_prototypes, aggregator=aggregator, init=init)
    if adapter == "TaskRes":
        return TaskRes(text_embeddings=text_prototypes, aggregator=aggregator)
    if adapter == "CLIPAdapter":
        return CLIPAdapter(text_embeddings=text_prototypes, aggregator=aggregator)
    if adapter == "TIPAdapter":
        return TIPAdapter(text_embeddings=text_prototypes, train_data=train_data,
                          train_labels=train_labels, aggregator=aggregator)
    raise ValueError(f"unknown adapter {adapter}")


def score_slides(model, data, device):
    """Slide-level P(Tumor) for each bag. The quantity validate_model() throws away.

    Returns the softmax probability rather than the raw margin only because it is
    bounded and easier to eyeball; every metric downstream is rank-based, so any
    monotone function of the Tumor-vs-Normal margin gives identical results.
    """
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for bag in data:
            logits = model(torch.tensor(bag, dtype=torch.float32, device=device))[0]
            out.append(float(logits.softmax(dim=0)[1].item()))   # classes[1] == "Tumor"
    return np.array(out)


def self_test():
    M.self_test()
    print("self_test: analyze_tumor_ratio_miladapter")
    # the split helper used for reporting must agree with utils.utils.fewshot_sampling;
    # on the cluster that is cross-checked against the real function on live data (see
    # the assertion in main()), here we check the contract it has to satisfy
    Y = np.array([0] * 239 + [1] * 160)
    tr, va = M.fewshot_split_ids(Y, 16, 0)
    assert len(tr) == 32 and len(va) == len(Y) - 32
    print(f"  split contract holds ({len(tr)} support / {len(va)} eval)")
    print("self_test: OK")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", help="MIL-Adapter data root: folder/project/encoder/*.npy")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--adapter", default="TaskRes",
                   choices=["ZSMIL", "TaskRes", "CLIPAdapter", "TIPAdapter"])
    p.add_argument("--aggregator", default="ABMIL",
                   choices=["BGAP", "BGMP", "ABMIL", "TransMIL", "WIKGMIL", "ILRAMIL", "RRTMIL"])
    p.add_argument("--init", choices=["ZS", "random"], default="random",
                   help="Only used by --adapter ZSMIL.")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--csv_path", default=None,
                   help="Project CSV with tumor_patch_pct and n_patches. "
                        "Defaults to ./local_data/csv/<project>.csv.")
    p.add_argument("--k_shots", type=int, default=16, choices=[2, 4, 8, 16])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="Seeds to train and evaluate. All of them are scored on the "
                        "SAME eval slides only if the split is seed-dependent -- it is, "
                        "so each seed is summarised on its own split and the seed spread "
                        "is reported separately from the slide bootstrap.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bin_edges", type=float, nargs="+", default=M.DEFAULT_EDGES)
    p.add_argument("--wall_pct", type=float, default=None)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--out_dir", default="./analysis_auc")
    p.add_argument("--out_prefix", default=None)
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.folder:
        p.error("--folder is required (or use --self_test)")

    import torch
    from tqdm import tqdm
    from utils.trainer import train_model
    from utils.utils import set_random_seeds, load_data, fewshot_sampling

    os.makedirs(args.out_dir, exist_ok=True)
    prefix = args.out_prefix or (f"{args.project}_{args.encoder}_{args.adapter}_"
                                 f"{args.aggregator}_k{args.k_shots}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    csv_path = args.csv_path or f"./local_data/csv/{args.project}.csv"
    csv_df = pd.read_csv(csv_path)
    if "tumor_patch_pct" not in csv_df.columns:
        raise SystemExit(f"{csv_path} has no tumor_patch_pct. Run add_tumor_pct_to_csv.py first.")
    pct_of = dict(zip(csv_df["WSI"], csv_df["tumor_patch_pct"]))
    wall = (args.wall_pct if args.wall_pct is not None
            else M.detection_wall_pct(csv_df["n_patches"]) if "n_patches" in csv_df
            else np.nan)

    classes, classes_id = get_project_data_camelyon16()
    text_prototypes = np.load(args.text_prototypes)
    X, Y, WSI = load_data(folder=args.folder, project=args.project,
                          encoder=args.encoder, classes=classes)
    missing = [s for s in WSI if s not in pct_of]
    if missing:
        raise SystemExit(f"{len(missing)} slide(s) lack tumor_patch_pct (e.g. {missing[:5]}). "
                         "Re-run add_tumor_pct_to_csv.py.")

    runs, per_slide_rows = [], []
    for seed in args.seeds:
        set_random_seeds(seed_value=seed)
        train_data, val_data, train_labels, val_labels = fewshot_sampling(
            X=X, Y=Y, k_shots=args.k_shots, seed=seed)

        # Recover slide indices by object identity: fewshot_sampling returns the SAME
        # array objects it was given (X[i] for i in ids), so this is exact and does not
        # duplicate its sampling logic. Approach taken from analyze_tumor_pct_accuracy.py.
        id_to_idx = {id(x): i for i, x in enumerate(X)}
        val_ids = np.array([id_to_idx[id(v)] for v in val_data])
        train_ids = np.array([id_to_idx[id(v)] for v in train_data])
        # ... and cross-check the feature-free reproduction MI-Zero relies on, so the
        # two entry points provably evaluate the same slides.
        tr_ref, va_ref = M.fewshot_split_ids(Y, args.k_shots, seed)
        assert np.array_equal(np.sort(train_ids), np.sort(tr_ref)), (
            "fewshot_split_ids disagrees with utils.utils.fewshot_sampling -- the "
            "MI-Zero arm would be evaluated on a different split")
        assert np.array_equal(np.sort(val_ids), np.sort(va_ref))

        model = build_model(args.adapter, text_prototypes, args.aggregator, args.init,
                            train_data, train_labels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                      weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        train_model(model, optimizer, criterion, scheduler, train_data, train_labels,
                    args.epochs)

        scores = score_slides(model, val_data, device)
        ids = np.array([WSI[i] for i in val_ids])
        pis = np.array([pct_of[s] for s in ids], dtype=float)
        is_tumor = np.array(val_labels) == classes.index("Tumor")

        runs.append((f"seed{seed}", scores[is_tumor], pis[is_tumor], scores[~is_tumor]))
        for s, sc, pi, lab in zip(ids, scores, pis, val_labels):
            per_slide_rows.append({"slide_id": s, "split": "val", "seed": seed,
                                   "gt_label": classes_id[int(lab)],
                                   "pred_label": classes_id[int(sc > 0.5)],
                                   "score": sc, "tumor_patch_pct": pi})
        print(f"[seed {seed}] {is_tumor.sum()} tumor / {(~is_tumor).sum()} normal eval "
              f"slides, AUC {M.auc_from_placement(M.placement_values(scores[is_tumor], scores[~is_tumor])):.4f}")

    # Seeds use different few-shot draws, so their eval sets differ slightly. The paired
    # bootstrap requires one shared slide set, so it runs on the intersection; the
    # per-seed point estimates above use each seed's own full eval set.
    common = set.intersection(*[set(r["slide_id"] for r in per_slide_rows
                                    if r["seed"] == s) for s in args.seeds])
    ps_df = pd.DataFrame(per_slide_rows)
    runs_common = []
    for seed in args.seeds:
        g = ps_df[(ps_df.seed == seed) & (ps_df.slide_id.isin(common))].sort_values("slide_id")
        t, n = g[g.gt_label == "Tumor"], g[g.gt_label == "Normal"]
        runs_common.append((f"seed{seed}", t["score"].to_numpy(),
                            t["tumor_patch_pct"].to_numpy(float), n["score"].to_numpy()))

    bins_df, point, seed_sd, boot, summary = M.summarise(
        runs_common, args.bin_edges, wall, args.n_boot)
    summary.update(adapter=args.adapter, aggregator=args.aggregator, encoder=args.encoder,
                   k_shots=args.k_shots, seeds=args.seeds, epochs=args.epochs, lr=args.lr,
                   text_prototypes=args.text_prototypes,
                   n_slides_common_to_all_seeds=len(common))

    ps_path = os.path.join(args.out_dir, f"{prefix}_per_slide.csv")
    bins_path = os.path.join(args.out_dir, f"{prefix}_bin_auc.csv")
    json_path = os.path.join(args.out_dir, f"{prefix}_summary.json")
    png = os.path.join(args.out_dir, f"{prefix}_auc_vs_tumor_pct.png")
    ps_df.to_csv(ps_path, index=False)
    bins_df.to_csv(bins_path, index=False)
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    M.make_figure(bins_df.iloc[:-1], runs_common, args.bin_edges, wall, png,
                  subtitle=f"{args.adapter}+{args.aggregator}, k={args.k_shots}, "
                           f"{len(args.seeds)} seeds; band = 95% paired bootstrap over slides")

    M.print_report(f"{args.adapter}+{args.aggregator} k={args.k_shots}", bins_df, summary)
    print(f"\nper-slide -> {ps_path}\nbins -> {bins_path}\nsummary -> {json_path}\nfigure -> {png}")
    print(f"\nBootstrap ran on the {len(common)} slides common to all {len(args.seeds)} "
          f"seeds (few-shot draws differ by seed).\nCompare auc_gap and the slope against "
          f"analyze_tumor_ratio_mizero.py: same metric code, same eval slides.")


if __name__ == "__main__":
    main()
