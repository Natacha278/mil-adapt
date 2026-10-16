"""
run_text_topk_mil.py

Trains and evaluates Text-TopK MIL (utils/text_topk_mil.py, steps 1-3 of the method) on
the few-shot CAMELYON16 splits, then compares it with other methods by witness rate and
lesion stage via compare_aggregation_vs_witness_rate.py.

For every seed: the support set is the same k-per-class draw as main.py / MIL-Adapter
(utils.tumor_ratio_metrics.fewshot_split_ids, checked against fewshot_sampling in
analyze_tumor_ratio_miladapter.py); the model is trained on it and every eval slide is
scored. Prediction (--threshold, several allowed):
  zero       Tumor iff S > 0 (steps 1-3, the trained bias sets the cut)
  (--neg_weight > 0, torch only: adds the normal-tail ranking loss -- top-M patches of every
   normal support slide as hard negatives; see utils/text_topk_mil_torch.py)
  conformal  step 4: leave-one-normal-out conformal threshold (utils/conformal.py) at a
             target false-positive rate --alpha on normal slides; one extra fit per
             support normal (16 at k=16); one CSV per --alpha value from the same fits,
             ..._conf<alpha>_per_slide.csv, with
             slide_score = margin (mean S_-j(x) - z_j), score = sigmoid(margin), p_value.

One configuration per (K, lambda) pair given with --topk / --lam. Each writes
  <out_dir>/TextTopK_K<K>_lam<lam>_per_slide.csv   (slide_id, split, seed, gt_label,
      pred_label, score = P(Tumor), slide_score = S)
  <out_dir>/TextTopK_K<K>_lam<lam>_training.csv    (per seed: ||Delta||, cos to the text
      direction, bias, final loss, support accuracy; conformal: the held-out normal
      scores z_j and the in-sample optimism = mean(z_j - S_full(normal j)))
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
import itertools
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import tumor_ratio_metrics as TRM              # noqa: E402
from utils.text_topk_mil import TextTopKMIL, sigmoid     # noqa: E402
from utils.conformal import ConformalDetector, count_max  # noqa: E402

CLASSES = ["Normal", "Tumor"]


def load_features(folder, project, encoder, wsis):
    """Same files as utils.utils.load_data, without importing torch."""
    return [np.load(os.path.join(folder, project, encoder, f"{w}.npy")).astype(np.float32)
            for w in tqdm(wsis, desc="loading features")]


def label_of(k, lam, rule="zero", alpha=None, nw=0.0, m=None, margin=None):
    extra = f", tail-rank {nw:g} (M={m}, margin {margin:g})" if nw > 0 else ""
    extra += f", conformal α={alpha:g}" if rule == "conformal" else ""
    return f"Text-TopK (K={k:g}, λ={lam:g}{extra})"


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
    p.add_argument("--concepts", default=None,
                   help="Concept file from encode_concepts.py: multi-concept scorer (torch backend). "
                        "--text_prototypes is then unused.")
    p.add_argument("--tau", type=float, default=1.0, help="Multi-concept smooth-max temperature (logits).")
    p.add_argument("--lam_leaf", type=float, default=None,
                   help="Multi-concept: per-concept residual prior; enables shared side residuals (prior --lam).")
    p.add_argument("--neg_weight", type=float, nargs="+", default=[0.0],
                   help="Weight(s) of the normal-tail ranking loss (torch backend; 0 = off).")
    p.add_argument("--neg_m", type=int, default=32,
                   help="Ranking loss: top-M patches of every normal support slide as hard negatives.")
    p.add_argument("--neg_margin", type=float, default=10.0,
                   help="Ranking loss margin, in logit units (scale ~56, so 10 ~ 0.18 in cosine).")
    p.add_argument("--threshold", nargs="+", choices=["zero", "conformal"], default=["zero"],
                   help="Decision rule(s); each gets its own per-slide CSV.")
    p.add_argument("--alpha", type=float, nargs="+", default=[0.1],
                   help="Conformal: target false-positive rate(s) on normal slides (>= 1/(k_shots+1)). "
                        "The LOO fits do not depend on alpha, so several cost nothing extra.")
    p.add_argument("--baseline", nargs="*", default=[], help='"LABEL=path" per-slide CSVs.')
    p.add_argument("--lesion_csv", default=None)
    p.add_argument("--xml_dir", default=None)
    p.add_argument("--h5_dir", default=None)
    p.add_argument("--calibrate_zero_shot", action="store_true",
                   help="Passed to the comparison (zero-shot baselines only).")
    p.add_argument("--no_compare", action="store_true")
    p.add_argument("--out_dir", default="analysis/analysis_text_topk")
    args = p.parse_args()
    if any(w > 0 for w in args.neg_weight) and args.backend != "torch":
        p.error("--neg_weight > 0 needs --backend torch (the ranking loss is torch-only)")
    if args.concepts and args.backend != "torch":
        p.error("--concepts needs --backend torch")

    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(args.csv_path)
    wsis = meta.WSI.to_numpy()
    Y = np.array([CLASSES.index(g) for g in meta.GT])
    X = load_features(args.folder, args.project, args.encoder, wsis)
    protos = None if args.concepts else np.load(args.text_prototypes)
    concept_tag = ""
    if args.concepts:
        from utils.text_topk_mil_torch import load_concepts
        _, cn, ct = load_concepts(args.concepts)
        concept_tag = (f"{int(ct.sum())}T{int((~ct).sum())}N_tau{args.tau:g}"
                       + (f"_leaf{args.lam_leaf:g}" if args.lam_leaf is not None else ""))
        print(f"multi-concept scorer: tumor {[n for n, t in zip(cn, ct) if t]}, "
              f"normal {[n for n, t in zip(cn, ct) if not t]}")

    if args.backend == "torch":
        import torch
        from utils.text_topk_mil_torch import make_text_topk
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_and_fit(k, lam, bags, labels, nw=0.0):
        if args.backend == "numpy":
            return TextTopKMIL(protos, k=k, lam=lam).fit(bags, labels, epochs=args.epochs,
                                                         lr=args.lr, lr_b=args.lr_b)
        m = make_text_topk(protos, concepts=args.concepts, k=k, lam=lam, tau=args.tau,
                           lam_leaf=args.lam_leaf).to(device)
        return m.fit(bags, labels, epochs=args.epochs, lr=args.lr, lr_b=args.lr_b,
                     neg_weight=nw, neg_m=args.neg_m, neg_margin=args.neg_margin)

    def slide_score(model, H):
        if args.backend == "numpy":
            return model.slide_score(H)
        with torch.no_grad():
            return float(model.slide_score(torch.as_tensor(H, device=device))[0])

    if "conformal" in args.threshold:
        rules = {}
        for a in args.alpha:                                 # fail before any training
            rules.setdefault(count_max(args.k_shots, a), []).append(a)
        for c, al in sorted(rules.items()):
            print(f"conformal alpha {', '.join(f'{v:g}' for v in al)}: Tumor iff the slide beats "
                  f"at least {args.k_shots - c} of the {args.k_shots} held-out normals "
                  f"(nominal FPR {(c + 1) / (args.k_shots + 1):.3f})"
                  + ("  [same rule]" if len(al) > 1 else ""))

    produced = []
    for k, lam, nw in itertools.product(args.topk, args.lam, args.neg_weight):
        rows = {r: [] for r in args.threshold}
        train_rows = []
        for seed in args.seeds:
            tr, va = TRM.fewshot_split_ids(Y, args.k_shots, seed)
            bags_tr = [X[i] for i in tr]
            model = make_and_fit(k, lam, bags_tr, Y[tr], nw)
            s_tr = np.array([slide_score(model, H) for H in bags_tr])
            info = {"seed": seed, **model.summary(),
                    "support_accuracy": float(((s_tr > 0) == (Y[tr] == 1)).mean())}
            det = None
            if "conformal" in args.threshold:
                det = ConformalDetector.fit(bags_tr, Y[tr],
                                            lambda b, y: make_and_fit(k, lam, b, y, nw),
                                            slide_score, alpha=max(args.alpha))
                in_sample = s_tr[Y[tr] == 0]          # same normal order as det.z
                info.update({**{f"conf_{kk}": v for kk, v in det.summary().items()
                                if kk in ("n_normals", "z_mean", "z_max")},
                             "conf_optimism": float(np.mean(det.z - in_sample)),
                             "conf_z": " ".join(f"{v:.4f}" for v in det.z)})
            train_rows.append(info)
            for i in va:
                S = slide_score(model, X[i])
                base = {"slide_id": wsis[i], "split": "val", "seed": seed,
                        "gt_label": CLASSES[Y[i]]}
                if "zero" in rows:
                    rows["zero"].append({**base, "pred_label": CLASSES[int(S > 0)],
                                         "score": float(sigmoid(S)), "slide_score": S})
                if det is not None:
                    e = det.evaluate(X[i])
                    rows["conformal"].append({**base, "score": float(sigmoid(e["margin"])),
                                              "slide_score": e["margin"],
                                              "p_value": e["p_value"], "count": e["count"],
                                              "S_full": S})
        tag = (f"TextTopK_K{k:g}_lam{lam:g}"
               + (f"_neg{nw:g}m{args.neg_m}mg{args.neg_margin:g}" if nw > 0 else "")
               + (f"_mc{concept_tag}" if args.concepts else "")
               + ("_torch" if args.backend == "torch" else ""))
        tdf = pd.DataFrame(train_rows)
        tdf.to_csv(os.path.join(args.out_dir, f"{tag}_training.csv"), index=False)
        outputs = [("zero", None, pd.DataFrame(rows["zero"]))] if "zero" in rows else []
        for a in (args.alpha if "conformal" in rows else []):
            df = pd.DataFrame(rows["conformal"])
            c = count_max(args.k_shots, a)                # same rule as det.predict
            df.insert(4, "pred_label", np.where(df["count"] <= c, "Tumor", "Normal"))
            outputs.append(("conformal", a, df))
        for rule, a, df in outputs:
            rtag = tag + (f"_conf{a:g}" if rule == "conformal" else "")
            path = os.path.join(args.out_dir, f"{rtag}_per_slide.csv")
            df.to_csv(path, index=False)
            g = df.groupby("seed")
            acc = g.apply(lambda d: (d.gt_label == d.pred_label).mean(), include_groups=False)
            spec = g.apply(lambda d: (d[d.gt_label == "Normal"].pred_label == "Normal").mean(),
                           include_groups=False)
            sens = g.apply(lambda d: (d[d.gt_label == "Tumor"].pred_label == "Tumor").mean(),
                           include_groups=False)
            lab = label_of(k, lam, rule, a, nw, args.neg_m, args.neg_margin)
            if args.concepts:
                lab = lab.replace("Text-TopK (", f"Text-TopK multi-concept[{concept_tag}] (")
            msg = (f"{lab}: eval accuracy {acc.mean():.3f} ± {acc.std():.3f}, "
                   f"sensitivity {sens.mean():.3f}, specificity {spec.mean():.3f}")
            if rule == "conformal":
                msg += (f" (target >= {1 - (count_max(args.k_shots, a) + 1) / (args.k_shots + 1):.3f}); in-sample "
                        f"optimism of the support normals {tdf.conf_optimism.mean():+.3f}")
            else:
                msg += (f"; support accuracy {tdf.support_accuracy.mean():.3f}; ||Delta|| "
                        f"{tdf.delta_norm.mean():.3f}, cos to text {tdf.cos_to_text.mean():.3f}")
            print(msg)
            produced.append(f"{lab}={path}")

    if args.no_compare:
        return
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
