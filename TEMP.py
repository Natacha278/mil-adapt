"""
run_text_topk_mil.py

Trains and evaluates Text-TopK MIL (utils/text_topk_mil.py, steps 1-3 of the method) on
the few-shot CAMELYON16 splits, then compares it with other methods by witness rate and
lesion stage via compare_aggregation_vs_witness_rate.py.

For every seed: the support set is the same k-per-class draw as main.py / MIL-Adapter
(utils.tumor_ratio_metrics.fewshot_split_ids, checked against fewshot_sampling in
analyze_tumor_ratio_miladapter.py); the model is trained on it and every eval slide is
scored. Prediction: Tumor iff S > 0 (no step-4 calibration).

One configuration per (K, lambda) pair given with --topk / --lam. Each writes
  <out_dir>/TextTopK_K<K>_lam<lam>_per_slide.csv   (slide_id, split, seed, gt_label,
      pred_label, score = P(Tumor), slide_score = S)
  <out_dir>/TextTopK_K<K>_lam<lam>_training.csv    (per seed: ||Delta||, cos to the text
      direction, bias, final loss, support accuracy)
in the per-slide format the comparison script reads. The comparison is then run with any
--baseline "LABEL=path" CSVs (e.g. the existing MIL-Adapter and MI-Zero outputs).

--backend numpy (default, CPU) uses utils/text_topk_mil.py; --backend torch uses the
PyTorch version utils/text_topk_mil_torch.py (GPU if available), which gives the same
parameters and is the one to extend (tail_quantile_loss, mimic prompts). Loads the
MIL-format .npy features like utils.utils.load_data.

    python run_text_topk_mil.py \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --topk 1 3 --lam 0.01 0.1 1 --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --baseline "MIL-Adapter (TaskRes+ABMIL)=analysis/analysis_auc/CAMELYON16_CONCH_TaskRes_ABMIL_k16_per_slide.csv" \
                   "MI-Zero top-1=analysis/analysis_mizero/CAMELYON16_MIZERO_logit_k1_per_slide.csv" \
        --lesion_csv analysis/analysis_lesion_size_descr/CAMELYON16_CONCH_zs_per_lesion.csv \
        --calibrate_zero_shot --out_dir analysis/analysis_text_topk
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import tumor_ratio_metrics as TRM              # noqa: E402
from utils.text_topk_mil import TextTopKMIL, sigmoid     # noqa: E402

CLASSES = ["Normal", "Tumor"]


def load_features(folder, project, encoder, wsis):
    """Same files as utils.utils.load_data, without importing torch."""
    return [np.load(os.path.join(folder, project, encoder, f"{w}.npy")).astype(np.float32)
            for w in tqdm(wsis, desc="loading features")]


def label_of(k, lam):
    return f"Text-TopK (K={k:g}, λ={lam:g})"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folder", required=True)
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--csv_path", default="./local_data/csv/CAMELYON16.csv")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--k_shots", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--topk", type=float, nargs="+", default=[1],
                   help="K of the top-K slide score (K < 1: fraction of the slide's patches).")
    p.add_argument("--lam", type=float, nargs="+", default=[0.1],
                   help="Weight of ||Delta||^2 (pull toward the text direction).")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam step for Delta.")
    p.add_argument("--lr_b", type=float, default=5e-2, help="Adam step for the bias.")
    p.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    p.add_argument("--baseline", nargs="*", default=[], help='"LABEL=path" per-slide CSVs.')
    p.add_argument("--lesion_csv", default=None)
    p.add_argument("--xml_dir", default=None)
    p.add_argument("--h5_dir", default=None)
    p.add_argument("--calibrate_zero_shot", action="store_true",
                   help="Passed to the comparison (zero-shot baselines only).")
    p.add_argument("--no_compare", action="store_true")
    p.add_argument("--out_dir", default="analysis/analysis_text_topk")
    args = p.parse_args()

    # os.makedirs(args.out_dir, exist_ok=True)
    # meta = pd.read_csv(args.csv_path)
    # wsis = meta.WSI.to_numpy()
    # Y = np.array([CLASSES.index(g) for g in meta.GT])
    # X = load_features(args.folder, args.project, args.encoder, wsis)
    # protos = np.load(args.text_prototypes)

    # if args.backend == "torch":
    #     import torch
    #     from utils.text_topk_mil_torch import TextTopKMIL as TorchTextTopK
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # def make_and_fit(k, lam, bags, labels):
    #     if args.backend == "numpy":
    #         return TextTopKMIL(protos, k=k, lam=lam).fit(bags, labels, epochs=args.epochs,
    #                                                      lr=args.lr, lr_b=args.lr_b)
    #     m = TorchTextTopK(protos, k=k, lam=lam).to(device)
    #     return m.fit(bags, labels, epochs=args.epochs, lr=args.lr, lr_b=args.lr_b)

    # def slide_score(model, H):
    #     if args.backend == "numpy":
    #         return model.slide_score(H)
    #     with torch.no_grad():
    #         return float(model.slide_score(torch.as_tensor(H, device=device))[0])

    produced = []
    for k in args.topk:
        for lam in args.lam:
            rows, train_rows = [], []
            # for seed in args.seeds:
            #     tr, va = TRM.fewshot_split_ids(Y, args.k_shots, seed)
            #     model = make_and_fit(k, lam, [X[i] for i in tr], Y[tr])
            #     s_tr = np.array([slide_score(model, X[i]) for i in tr])
            #     train_rows.append({"seed": seed, **model.summary(),
            #                        "support_accuracy": float(((s_tr > 0) == (Y[tr] == 1)).mean())})
            #     for i in va:
            #         S = slide_score(model, X[i])
            #         rows.append({"slide_id": wsis[i], "split": "val", "seed": seed,
            #                      "gt_label": CLASSES[Y[i]], "pred_label": CLASSES[int(S > 0)],
            #                      "score": float(sigmoid(S)), "slide_score": S})
            # df = pd.DataFrame(rows)
            tag = f"TextTopK_K{k:g}_lam{lam:g}" + ("_torch" if args.backend == "torch" else "")
            path = os.path.join(args.out_dir, f"{tag}_per_slide.csv")
            # df.to_csv(path, index=False)
            # tdf = pd.DataFrame(train_rows)
            # tdf.to_csv(os.path.join(args.out_dir, f"{tag}_training.csv"), index=False)
            # acc = df.groupby("seed").apply(lambda g: (g.gt_label == g.pred_label).mean(),
            #                                include_groups=False)
            # print(f"{label_of(k, lam)}: eval accuracy {acc.mean():.3f} ± {acc.std():.3f}; "
            #       f"support accuracy {tdf.support_accuracy.mean():.3f}; "
            #       f"||Delta|| {tdf.delta_norm.mean():.3f}, cos to text {tdf.cos_to_text.mean():.3f}")
            produced.append(f"{label_of(k, lam)}={path}")

    # if args.no_compare:
    #     return
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "compare_aggregation_vs_witness_rate.py"),
           "--csv_path", args.csv_path, "--k_shots", str(args.k_shots),
           "--seeds", *map(str, args.seeds),
           "--per_slide", *produced, *args.baseline,
           "--out_prefix", os.path.join(args.out_dir, "compare")]
    if args.lesion_csv:
        cmd += ["--lesion_csv", args.lesion_csv]
    if args.xml_dir:
        cmd += ["--xml_dir", args.xml_dir]
    if args.h5_dir:
        cmd += ["--h5_dir", args.h5_dir]
    if args.calibrate_zero_shot:
        cmd += ["--calibrate_zero_shot"]
    print("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
