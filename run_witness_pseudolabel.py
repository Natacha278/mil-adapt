"""
run_witness_pseudolabel.py

Few-shot CAMELYON16 runner for Text-TopK MIL + witness-rate-aware pseudo-labelling
(utils/witness_pseudolabel.py). Same splits, prediction rule (top-K slide score, S > 0) and
per-slide CSV format as run_text_topk_mil.py, so the outputs drop into
compare_aggregation_vs_witness_rate.py next to the existing Text-TopK runs.

Per seed:
  round 0  n_hat_j from zero-shot patch p-values vs all support-normal patches;
           pseudo-positives = top n_hat_j zero-shot patches of each support tumor slide
  round r  refit (slide BCE + text prior + tail-rank + patch BCE on pseudo-positives and on
           every support-normal patch); re-estimate n_hat_j with 2-fold cross-fitted models
           (null = the held-out half of the normals), reselect with the round-r model
One configuration per (rounds, patch_weight) pair given with --rounds / --patch_weight.

OUTPUTS (<out_dir>/<tag>_...)
  _per_slide.csv     slide_id, split, seed, gt_label, pred_label, score = P(Tumor), slide_score
  _training.csv      per seed: ||Delta||, cos to text, bias, support accuracy, median n_hat
  _pseudolabels.csv  per seed, round and support tumor slide: n_patches, n_hat, pi_hat and,
                     with --xml_dir/--h5_dir, n_true, precision / recall of the pseudo-positives
                     and precision of a fixed top-3 selection (reference)

    python run_witness_pseudolabel.py --self_test
    python run_witness_pseudolabel.py \\
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \\
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \\
        --topk 3 --lam 0.1 --neg_weight 1 --neg_m 32 --neg_margin 10 \\
        --rounds 1 3 --patch_weight 0.3 1 --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --xml_dir $SCRATCH/camelyon16/annotations \\
        --baseline "Text-TopK (K=3, λ=0.1, tail-rank 1)=analysis/analysis_text_topk/TextTopK_K3_lam0.1_neg1m32mg10_torch_per_slide.csv" \\
        --out_dir analysis/analysis_pseudolabel
"""

import argparse
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_text_topk_mil import CLASSES, load_features   # noqa: E402

DEFAULT_H5 = "/project/rrg-josedolz/natgill/data/camelyon16/trident/20x_512px_0px_overlap/features_conch_v1"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--csv_path", default="./local_data/csv/CAMELYON16.csv")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--concepts", default=None, help="Concept file (encode_concepts.py): multi-concept scorer.")
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--lam_leaf", type=float, default=None)
    p.add_argument("--k_shots", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--topk", type=float, default=3)
    p.add_argument("--lam", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_b", type=float, default=5e-2)
    p.add_argument("--neg_weight", type=float, default=1.0, help="Normal-tail ranking loss weight (0 = off).")
    p.add_argument("--neg_m", type=int, default=32)
    p.add_argument("--neg_margin", type=float, default=10.0)
    p.add_argument("--rounds", type=int, nargs="+", default=[3],
                   help="Pseudo-labelling rounds (1 = zero-shot estimate only, no re-estimation).")
    p.add_argument("--patch_weight", type=float, nargs="+", default=[1.0], help="Weight of the patch BCE.")
    p.add_argument("--z", type=float, default=3.0, help="Excess-count margin in binomial SDs.")
    p.add_argument("--t_max", type=float, default=0.05, help="Largest p-value level of the excess count.")
    p.add_argument("--cap_frac", type=float, default=0.5, help="n_hat <= cap_frac * patches of the slide.")
    p.add_argument("--no_crossfit", action="store_true",
                   help="Re-estimate with the in-sample null (optimistic; ablation only).")
    p.add_argument("--xml_dir", default=None, help="With --h5_dir: pseudo-label precision diagnostics.")
    p.add_argument("--h5_dir", default=DEFAULT_H5)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--baseline", nargs="*", default=[], help='"LABEL=path" per-slide CSVs.')
    p.add_argument("--lesion_csv", default=None)
    p.add_argument("--no_compare", action="store_true")
    p.add_argument("--out_dir", default="analysis/analysis_pseudolabel")
    p.add_argument("--self_test", action="store_true")
    return p, p.parse_args(argv)


def run(args):
    import torch
    from utils import tumor_ratio_metrics as TRM
    from utils.text_topk_mil_torch import make_text_topk
    from utils.witness_pseudolabel import fit_witness_pseudolabel

    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(args.csv_path)
    wsis = meta.WSI.to_numpy()
    Y = np.array([CLASSES.index(g) for g in meta.GT])
    X = load_features(args.folder, args.project, args.encoder, wsis)
    protos = None if args.concepts else np.load(args.text_prototypes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gt = {}
    if args.xml_dir:
        from analyze_text_topk_vs_witness_rate import ground_truth
        from utils import witness_rate as W
        cache, skipped = ground_truth(wsis, dict(zip(wsis, meta.GT)), X, args.h5_dir, args.xml_dir,
                                      "centre", 0.5, args.mpp_level0 or W.DEFAULT_MPP_LEVEL0)
        gt = {i: cache[w]["gt"] for i, w in enumerate(wsis) if w in cache}
        print(f"patch ground truth for {len(gt)} tumor slides ({len(skipped)} without h5/xml)")

    def new_model():
        return make_text_topk(protos, concepts=args.concepts, k=args.topk, lam=args.lam, tau=args.tau,
                              lam_leaf=args.lam_leaf).to(device)

    fit_kw = {"epochs": args.epochs, "lr": args.lr, "lr_b": args.lr_b}
    if args.neg_weight > 0:
        fit_kw.update({"neg_weight": args.neg_weight, "neg_m": args.neg_m, "neg_margin": args.neg_margin})

    produced, summaries = [], []
    for R, eta in itertools.product(args.rounds, args.patch_weight):
        rows, train_rows, pl_rows = [], [], []
        for seed in args.seeds:
            tr, va = TRM.fewshot_split_ids(Y, args.k_shots, seed)
            bags = [X[i] for i in tr]
            gt_sup = {j: gt[i] for j, i in enumerate(tr) if i in gt}
            model, diag = fit_witness_pseudolabel(bags, Y[tr], new_model, rounds=R, fit_kw=fit_kw,
                                                  patch_weight=eta, z=args.z, t_max=args.t_max,
                                                  cap_frac=args.cap_frac, crossfit=not args.no_crossfit,
                                                  seed=seed, gt=gt_sup or None)

            def S_of(H):
                with torch.no_grad():
                    return float(model.slide_score(torch.as_tensor(H, device=device))[0])
            s_tr = np.array([S_of(H) for H in bags])
            d = pd.DataFrame(diag)
            d.insert(0, "seed", seed)
            d.insert(2, "slide_id", wsis[tr][d.slide.to_numpy()])
            pl_rows.append(d.drop(columns="slide"))
            last = d[d["round"] == d["round"].max()]
            train_rows.append({"seed": seed, **model.summary(),
                               "support_accuracy": float(((s_tr > 0) == (Y[tr] == 1)).mean()),
                               "n_hat_median": float(last.n_hat.median())})
            for i in va:
                S = S_of(X[i])
                rows.append({"slide_id": wsis[i], "split": "val", "seed": seed, "gt_label": CLASSES[Y[i]],
                             "pred_label": CLASSES[int(S > 0)], "score": float(sigmoid(S)), "slide_score": S})
        tag = (f"WRPseudo_K{args.topk:g}_lam{args.lam:g}_r{R}_eta{eta:g}_z{args.z:g}"
               + (f"_neg{args.neg_weight:g}m{args.neg_m}mg{args.neg_margin:g}" if args.neg_weight > 0 else "")
               + ("_nocf" if args.no_crossfit else "") + ("_mc" if args.concepts else ""))
        df, tdf, pdf = pd.DataFrame(rows), pd.DataFrame(train_rows), pd.concat(pl_rows, ignore_index=True)
        path = os.path.join(args.out_dir, f"{tag}_per_slide.csv")
        df.to_csv(path, index=False)
        tdf.to_csv(os.path.join(args.out_dir, f"{tag}_training.csv"), index=False)
        pdf.to_csv(os.path.join(args.out_dir, f"{tag}_pseudolabels.csv"), index=False)

        g = df.groupby("seed")
        acc = g.apply(lambda d: (d.gt_label == d.pred_label).mean(), include_groups=False)
        sens = g.apply(lambda d: (d[d.gt_label == "Tumor"].pred_label == "Tumor").mean(), include_groups=False)
        spec = g.apply(lambda d: (d[d.gt_label == "Normal"].pred_label == "Normal").mean(), include_groups=False)
        lab = (f"WR-pseudolabel (K={args.topk:g}, λ={args.lam:g}"
               + (f", tail-rank {args.neg_weight:g}" if args.neg_weight > 0 else "")
               + f", {R} round{'s' if R > 1 else ''}, η={eta:g})")
        print(f"{lab}: eval accuracy {acc.mean():.3f} ± {acc.std():.3f}, sensitivity {sens.mean():.3f}, "
              f"specificity {spec.mean():.3f}; support accuracy {tdf.support_accuracy.mean():.3f}; "
              f"median n_hat {tdf.n_hat_median.mean():.1f}")
        if "precision" in pdf:
            by_r = pdf.groupby("round")[["n_hat", "n_true", "precision", "precision_top3", "recall"]].mean()
            rho = pdf.groupby(["seed", "round"]).apply(
                lambda q: q.n_hat.corr(q.n_true, method="spearman"), include_groups=False).groupby("round").mean()
            by_r["spearman(n_hat, n_true)"] = rho
            print("  pseudo-labels on the support tumor slides (mean over seeds and slides):")
            print(by_r.to_string(float_format=lambda v: f"{v:.3f}"))
        produced.append(f"{lab}={path}")
        summaries.append({"rounds": R, "patch_weight": eta, "accuracy": acc.mean(),
                          "sensitivity": sens.mean(), "specificity": spec.mean(), "path": path})
    return produced, pd.DataFrame(summaries)


def compare(args, produced):
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "compare_aggregation_vs_witness_rate.py"),
           "--csv_path", args.csv_path, "--k_shots", str(args.k_shots), "--seeds", *map(str, args.seeds),
           "--per_slide", *produced, *args.baseline, "--out_prefix", os.path.join(args.out_dir, "compare")]
    if args.lesion_csv:
        cmd += ["--lesion_csv", args.lesion_csv]
    if args.xml_dir:
        cmd += ["--xml_dir", args.xml_dir, "--h5_dir", args.h5_dir]
    print("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def self_test():
    from analyze_text_topk_vs_witness_rate import make_synthetic_cohort
    root = make_synthetic_cohort(seed=11, n_slides=80, radii=(500, 800, 1500, 3000, 6000), n_mimic=6)
    out = os.path.join(root, "out")
    _, args = parse(["--folder", os.path.join(root, "feat"), "--csv_path", os.path.join(root, "C.csv"),
                     "--text_prototypes", os.path.join(root, "protos.npy"), "--h5_dir", os.path.join(root, "h5"),
                     "--xml_dir", os.path.join(root, "xml"), "--k_shots", "8", "--seeds", "0", "1",
                     "--topk", "1", "--epochs", "150", "--lr", "5e-3", "--rounds", "1", "3",
                     "--patch_weight", "1", "--out_dir", out])
    produced, summ = run(args)
    assert len(produced) == 2 and len(summ) == 2
    meta = pd.read_csv(args.csv_path)
    for p in summ.path:
        df = pd.read_csv(p)
        assert list(df.columns[:7]) == ["slide_id", "split", "seed", "gt_label", "pred_label", "score", "slide_score"]
        assert (df.pred_label == np.where(df.slide_score > 0, "Tumor", "Normal")).all()
        assert len(df) == 2 * (len(meta) - 16)
    pdf = pd.read_csv(summ.path.iloc[1].replace("_per_slide", "_pseudolabels"))
    assert set(pdf["round"]) == {0, 1, 2} and (pdf.n_hat >= 1).all() and pdf.slide_id.str.len().gt(0).all()
    last = pdf[pdf["round"] == 2]
    assert last.precision.mean() >= last.precision_top3.mean() - 0.05, last[["precision", "precision_top3"]].mean()
    print(summ.drop(columns="path").to_string(float_format=lambda v: f"{v:.3f}"))
    # baseline Text-TopK on the same splits, for reference
    _, b = parse(["--folder", args.folder, "--csv_path", args.csv_path, "--text_prototypes", args.text_prototypes,
                  "--k_shots", "8", "--seeds", "0", "1", "--topk", "1", "--epochs", "150", "--lr", "5e-3",
                  "--rounds", "1", "--patch_weight", "0", "--out_dir", out])
    _, sb = run(b)
    print(f"  reference (patch_weight 0 = Text-TopK + tail-rank): accuracy {sb.accuracy[0]:.3f}, "
          f"sensitivity {sb.sensitivity[0]:.3f}")
    print("self_test: OK")


def main():
    p, args = parse()
    if args.self_test:
        self_test()
        return
    if not args.folder:
        p.error("--folder is required (or --self_test)")
    produced, _ = run(args)
    if not args.no_compare:
        compare(args, produced)


if __name__ == "__main__":
    main()
