"""
run_concept_tail_mil.py

Few-shot CAMELYON16 runner for ConceptTail MIL (utils/concept_tail_mil.py): text-anchored,
subspace-constrained concept scorer + LSE / top-K pooling + pooled normal-tail pAUC loss,
with the guiding-proposition check (utils/tail_proposition.py).

Grid: one configuration per (--pool, --lam_tail, --extra_normals) triple.
  --pool          lse (learned gamma), top1, top3, ...   (ablation of K <= m)
  --lam_tail      weight of L_tail (0 = slide BCE + priors only)
  --extra_normals shot asymmetry: N extra normal slides added to the k-per-class support
                  (drawn per seed from the eval normals; the largest draw is removed from the
                  eval set of EVERY configuration of the run, so all share one eval set)

Same splits (utils.tumor_ratio_metrics.fewshot_split_ids), prediction rule (S > 0) and
per-slide CSV format as run_text_topk_mil.py; compare_aggregation_vs_witness_rate.py is
run at the end with any --baseline CSVs (it restricts every method to the common slides).

--reference_text_prototypes adds Text-TopK (K=--ref_topk, λ=--ref_lam, tail-rank
--ref_neg_weight) trained on the same supports, as an in-run reference.

Guiding proposition (with --xml_dir; patch GT of the eval tumor slides): for every method
(each configuration, the concept zero-shot scorer, the reference) and seed, TPR at the
operating point, patch pAUC on [0, beta*], patch AUC, and per tumor slide the predicted vs
observed probability of beating the median normal slide.

OUTPUTS (<out_dir>/)
  <tag>_per_slide.csv, <tag>_training.csv           per configuration
  proposition_per_slide.csv, proposition_by_bin.csv, proposition_scorer.csv,
  proposition_by_witness_rate.png                   (with --xml_dir)

    python run_concept_tail_mil.py --self_test
    python run_concept_tail_mil.py \\
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \\
        --concepts local_data/prompts/CONCH/CAMELYON16_concepttail.npz \\
        --bank local_data/prompts/CONCH/pathology_text_bank.npz --rank 16 \\
        --pool lse top1 top3 --lam_tail 0 1 --extra_normals 0 \\
        --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \\
        --reference_text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \\
        --xml_dir $SCRATCH/camelyon16/annotations \\
        --lesion_csv analysis/analysis_lesion_size_descr/CAMELYON16_CONCH_zs_per_lesion.csv \\
        --out_dir analysis/analysis_concept_tail
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
    p.add_argument("--concepts", help="encode_concepts.py .npz (tumor concepts T + normal look-alikes C).")
    p.add_argument("--bank", default=None, help="encode_text_bank.py .npz; without it B = span of the concepts.")
    p.add_argument("--rank", type=int, default=16, help="r, dimension of the text subspace B (0 = no a).")
    p.add_argument("--lam_a", type=float, default=0.1)
    p.add_argument("--lam_delta", type=float, default=None, help="Enables the free residual delta (strong L2).")
    p.add_argument("--pool", nargs="+", default=["lse"], help="lse, top1, top3, ...")
    p.add_argument("--tau", type=float, default=1.0, help="Initial concept smooth-max temperature (logits).")
    p.add_argument("--gamma", type=float, default=1.0, help="Initial LSE pooling temperature (logits).")
    p.add_argument("--no_learn_scale", action="store_true", help="Freeze alpha at the CONCH logit scale.")
    p.add_argument("--lam_scale", type=float, default=1.0, help="Prior on log alpha around CONCH's scale.")
    p.add_argument("--lam_tail", type=float, nargs="+", default=[1.0])
    p.add_argument("--tail_beta", type=float, default=None, help="Override beta (default K_eff / N_eff).")
    p.add_argument("--neff_factor", type=float, default=1.0,
                   help="N_eff = factor x median patches per normal slide (< 1: correlated patches).")
    p.add_argument("--mu", type=float, default=10.0, help="Tail-loss margin (logits).")
    p.add_argument("--kappa", type=float, default=1.0, help="Tail-loss temperature.")
    p.add_argument("--neg_weight", type=float, default=0.0, help="Ablation: per-slide top-M ranking loss.")
    p.add_argument("--neg_m", type=int, default=32)
    p.add_argument("--neg_margin", type=float, default=10.0)
    p.add_argument("--extra_normals", type=int, nargs="+", default=[0])
    p.add_argument("--k_shots", type=int, default=16)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam step for a (and delta).")
    p.add_argument("--lr_b", type=float, default=5e-2)
    p.add_argument("--lr_s", type=float, default=1e-2, help="Adam step for log alpha, log tau, log gamma.")
    p.add_argument("--reference_text_prototypes", default=None)
    p.add_argument("--ref_topk", type=float, default=3)
    p.add_argument("--ref_lam", type=float, default=0.1)
    p.add_argument("--ref_neg_weight", type=float, default=1.0)
    p.add_argument("--xml_dir", default=None)
    p.add_argument("--h5_dir", default=DEFAULT_H5)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--baseline", nargs="*", default=[], help='"LABEL=path" per-slide CSVs.')
    p.add_argument("--lesion_csv", default=None)
    p.add_argument("--no_compare", action="store_true")
    p.add_argument("--out_dir", default="analysis/analysis_concept_tail")
    p.add_argument("--self_test", action="store_true")
    return p, p.parse_args(argv)


def pool_spec(s):
    if s == "lse":
        return "lse", 1
    if s.startswith("top"):
        return "topk", float(s[3:])
    raise ValueError(f"--pool {s}: use lse or topK (e.g. top3)")


def run(args):
    import torch
    from utils import tumor_ratio_metrics as TRM
    from utils.concept_tail_mil import make_concept_tail
    from utils.tail_proposition import proposition_table
    from utils.text_topk_mil_torch import make_text_topk

    os.makedirs(args.out_dir, exist_ok=True)
    meta = pd.read_csv(args.csv_path)
    wsis = meta.WSI.to_numpy()
    Y = np.array([CLASSES.index(g) for g in meta.GT])
    X = load_features(args.folder, args.project, args.encoder, wsis)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for s in args.pool:
        pool_spec(s)

    gt = {}
    if args.xml_dir:
        from analyze_text_topk_vs_witness_rate import ground_truth
        from utils import witness_rate as W
        cache, skipped = ground_truth(wsis, dict(zip(wsis, meta.GT)), X, args.h5_dir, args.xml_dir,
                                      "centre", 0.5, args.mpp_level0 or W.DEFAULT_MPP_LEVEL0)
        gt = {i: cache[w] for i, w in enumerate(wsis) if w in cache}
        print(f"patch ground truth for {len(gt)} tumor slides ({len(skipped)} without h5/xml)")

    def new_model(pool):
        rule, k = pool_spec(pool)
        return make_concept_tail(args.concepts, bank=args.bank, r=args.rank, lam_a=args.lam_a,
                                 lam_delta=args.lam_delta, pool=rule, k=k, tau=args.tau, gamma=args.gamma,
                                 learn_scale=not args.no_learn_scale, lam_scale=args.lam_scale).to(device)

    fit_kw = {"epochs": args.epochs, "lr": args.lr, "lr_b": args.lr_b, "lr_s": args.lr_s,
              "tail_beta": args.tail_beta, "neff_factor": args.neff_factor, "mu": args.mu,
              "kappa": args.kappa, "neg_weight": args.neg_weight, "neg_m": args.neg_m,
              "neg_margin": args.neg_margin}

    @torch.no_grad()
    def pooled_score(model, s):
        """Slide score from numpy patch scores, on the model's device and dtype."""
        return float(model.pool(torch.as_tensor(s, dtype=model.w0.dtype, device=model.w0.device))[0])

    @torch.no_grad()
    def scores_of(fn, H):
        return fn(torch.as_tensor(H, device=device)).cpu().numpy().astype(np.float64)

    max_extra = max(args.extra_normals)
    splits = {}
    for seed in args.seeds:
        tr, va = TRM.fewshot_split_ids(Y, args.k_shots, seed)
        va_n = va[Y[va] == 0]
        extra = np.random.default_rng(7919 + seed).permutation(va_n)[:max_extra]
        if max_extra > len(va_n) - 2:
            raise SystemExit(f"--extra_normals {max_extra}: only {len(va_n)} normal eval slides")
        splits[seed] = (tr, np.setdiff1d(va, extra), extra)
    if max_extra:
        print(f"shot asymmetry: up to {max_extra} extra normal slides per seed, removed from every eval set")

    prop_rows, prop_info = [], []

    def proposition(method, seed, patch_fn, slide_fn, K, eva):
        if not gt:
            return
        nv = [i for i in eva if Y[i] == 0]
        tv = [i for i in eva if Y[i] == 1 and i in gt]
        ns = [scores_of(patch_fn, X[i]) for i in nv]
        ts = [scores_of(patch_fn, X[i]) for i in tv]
        df, info = proposition_table(ns, [slide_fn(i, s) for i, s in zip(nv, ns)], ts,
                                     [gt[i]["gt"] for i in tv], [slide_fn(i, s) for i, s in zip(tv, ts)],
                                     K=int(K), neff_factor=args.neff_factor,
                                     tumor_info=[{"slide_id": wsis[i], "tumor_area_pct": gt[i]["tumor_area_pct"],
                                                  "lesion_stage": gt[i]["lesion_stage"]} for i in tv])
        df.insert(0, "method", method)
        df.insert(1, "seed", seed)
        prop_rows.append(df)
        prop_info.append({"method": method, "seed": seed, **info})

    produced, summaries = [], []
    # ---- references (once per seed): concept zero-shot and Text-TopK -----------------------
    ref_rows = []
    for seed in args.seeds:
        tr, eva, extra = splits[seed]
        zs = new_model("top1")
        proposition("zero-shot concepts (max)", seed, zs.zero_shot_scores, lambda i, s: float(s.max()), 1, eva)
        if args.reference_text_prototypes:
            ref = make_text_topk(np.load(args.reference_text_prototypes), k=args.ref_topk,
                                 lam=args.ref_lam).to(device)
            kw = ({"neg_weight": args.ref_neg_weight, "neg_m": args.neg_m, "neg_margin": args.neg_margin}
                  if args.ref_neg_weight > 0 else {})
            ref.fit([X[i] for i in tr], Y[tr], epochs=args.epochs, lr=args.lr, lr_b=args.lr_b, **kw)
            K = ref._k(10 ** 6) if args.ref_topk >= 1 else 1
            for i in eva:
                S = float(ref.slide_score(torch.as_tensor(X[i], device=device))[0].detach())
                ref_rows.append({"slide_id": wsis[i], "split": "val", "seed": seed, "gt_label": CLASSES[Y[i]],
                                 "pred_label": CLASSES[int(S > 0)], "score": float(sigmoid(S)), "slide_score": S})
            proposition(f"Text-TopK K={args.ref_topk:g}", seed, ref.patch_scores,
                        lambda i, s, K=K: float(np.sort(s)[-K:].mean()), K, eva)
    if ref_rows:
        lab = (f"Text-TopK (K={args.ref_topk:g}, λ={args.ref_lam:g}"
               + (f", tail-rank {args.ref_neg_weight:g}" if args.ref_neg_weight > 0 else "") + ") [in-run]")
        path = os.path.join(args.out_dir, f"reference_TextTopK_K{args.ref_topk:g}_per_slide.csv")
        pd.DataFrame(ref_rows).to_csv(path, index=False)
        produced.append(f"{lab}={path}")
        summaries.append({"method": lab, **_rates(pd.DataFrame(ref_rows))})

    # ---- ConceptTail grid -------------------------------------------------------------------
    for pool, lt, xn in itertools.product(args.pool, args.lam_tail, args.extra_normals):
        rows, train_rows = [], []
        for seed in args.seeds:
            tr, eva, extra = splits[seed]
            sup = np.concatenate([tr, extra[:xn]])
            model = new_model(pool).fit([X[i] for i in sup], Y[sup], lam_tail=lt, **fit_kw)

            def S_of(H):
                with torch.no_grad():
                    return float(model.slide_score(torch.as_tensor(H, device=device))[0])
            s_tr = np.array([S_of(X[i]) for i in sup])
            train_rows.append({"seed": seed, "n_support_normals": int((Y[sup] == 0).sum()), **model.summary(),
                               "support_accuracy": float(((s_tr > 0) == (Y[sup] == 1)).mean())})
            for i in eva:
                S = S_of(X[i])
                rows.append({"slide_id": wsis[i], "split": "val", "seed": seed, "gt_label": CLASSES[Y[i]],
                             "pred_label": CLASSES[int(S > 0)], "score": float(sigmoid(S)), "slide_score": S})
            lab_p = f"ConceptTail {pool}, λ_tail={lt:g}" + (f", +{xn} normals" if xn else "")
            K = model.k_eff
            proposition(lab_p, seed, model.patch_scores,
                        (lambda i, s: pooled_score(model, s)) if pool == "lse"
                        else (lambda i, s, K=int(K): float(np.sort(s)[-K:].mean())), K, eva)
        tag = (f"ConceptTail_{pool}_r{args.rank}_tail{lt:g}" + (f"_x{xn}" if xn else "")
               + (f"_neff{args.neff_factor:g}" if args.neff_factor != 1 else "")
               + (f"_delta{args.lam_delta:g}" if args.lam_delta is not None else "")
               + (f"_neg{args.neg_weight:g}" if args.neg_weight > 0 else ""))
        df, tdf = pd.DataFrame(rows), pd.DataFrame(train_rows)
        path = os.path.join(args.out_dir, f"{tag}_per_slide.csv")
        df.to_csv(path, index=False)
        tdf.to_csv(os.path.join(args.out_dir, f"{tag}_training.csv"), index=False)
        lab = (f"ConceptTail (pool {pool}, r={args.rank}, λ_tail={lt:g}" + (f", +{xn} normals" if xn else "") + ")")
        r = _rates(df)
        print(f"{lab}: eval accuracy {r['accuracy']:.3f} ± {r['accuracy_sd']:.3f}, sensitivity "
              f"{r['sensitivity']:.3f}, specificity {r['specificity']:.3f}; J {tdf.tail_J.mean():.0f} "
              f"(beta {tdf.tail_beta.mean():.2e}); alpha {tdf.alpha.mean():.1f}, tau {tdf.tau.mean():.2f}"
              + (f", gamma {tdf.gamma.mean():.2f}" if pool == "lse" else "")
              + f"; ||a|| {tdf.a_norm.mean():.3f}, cos to text {tdf.cos_to_text.mean():.3f}")
        produced.append(f"{lab}={path}")
        summaries.append({"method": lab, **r, "path": path})

    summ = pd.DataFrame(summaries)
    if prop_rows:
        _proposition_report(pd.concat(prop_rows, ignore_index=True), pd.DataFrame(prop_info), args)
    return produced, summ


def _rates(df):
    g = df.groupby("seed")
    acc = g.apply(lambda d: (d.gt_label == d.pred_label).mean(), include_groups=False)
    sens = g.apply(lambda d: (d[d.gt_label == "Tumor"].pred_label == "Tumor").mean(), include_groups=False)
    spec = g.apply(lambda d: (d[d.gt_label == "Normal"].pred_label == "Normal").mean(), include_groups=False)
    return {"accuracy": acc.mean(), "accuracy_sd": acc.std(), "sensitivity": sens.mean(),
            "specificity": spec.mean()}


def _proposition_report(df, info, args):
    from analyze_text_topk_vs_witness_rate import WR_BINS, WR_LABELS
    from utils.tail_proposition import proposition_figure
    df["wr_bin"] = pd.cut(df.tumor_area_pct, WR_BINS, labels=WR_LABELS, include_lowest=True).astype(str)
    df.to_csv(os.path.join(args.out_dir, "proposition_per_slide.csv"), index=False)
    info.to_csv(os.path.join(args.out_dir, "proposition_scorer.csv"), index=False)
    cols = ["observed", "pred", "pred_simple", "pred_slide", "slide_auc"]
    ps = df.groupby(["method", "wr_bin", "seed"])[cols].mean()
    by = ps.groupby(["method", "wr_bin"]).mean().add_suffix("_mean").reset_index().rename(columns={"wr_bin": "bin"})
    by["n_per_seed"] = df.groupby(["method", "wr_bin", "seed"]).size().groupby(["method", "wr_bin"]).mean().values
    by.to_csv(os.path.join(args.out_dir, "proposition_by_bin.csv"), index=False)
    order = [b for b in WR_LABELS if b in set(by.bin)]
    proposition_figure(by, order, os.path.join(args.out_dir, "proposition_by_witness_rate.png"),
                       "Guiding proposition: predicted vs observed detection by witness rate",
                       footer=(f"Observed: tumor eval slides scoring above the median normal eval slide (mean over "
                               f"{info.seed.nunique()} seeds). Predicted: P(Poisson(λ_K) + Bin(m, TPR(β_K)) ≥ K), "
                               f"m = annotated tumor patches, TPR at the median normal-slide operating point "
                               f"β_K = λ_K/N_eff, N_eff = {args.neff_factor:g} × median patches per normal slide."))
    pd.set_option("display.width", 220)
    print("\n" + "=" * 100 + "\nguiding proposition -- scorer level (mean over seeds):")
    print(info.groupby("method")[["patch_auc", "pauc_beta", "tpr_beta", "tpr_beta_med", "beta"]].mean()
          .sort_values("pauc_beta").to_string(float_format=lambda v: f"{v:.4g}"))
    print("\nobserved vs predicted (refined) / predicted (original statement), by tumor-area bin:")
    for m, g in by.groupby("method", sort=False):
        g = g.set_index("bin").reindex(order)
        print(f"  {m}: " + "  ".join(f"{b} {o:.2f}/{p:.2f}/{q:.2f}" for b, o, p, q in
                                     zip(order, g.observed_mean, g.pred_mean, g.pred_simple_mean)))
    r = np.corrcoef(by.observed_mean, by.pred_mean)[0, 1] if len(by) > 2 else np.nan
    print(f"correlation observed vs predicted over (method, bin): {r:.3f}")
    print(f"-> {args.out_dir}/proposition_by_witness_rate.png, proposition_*.csv")


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
    rng = np.random.default_rng(0)
    P = np.load(os.path.join(root, "protos.npy"))                     # [D, 2] = (normal, tumor)
    np.savez(os.path.join(root, "concepts.npz"), protos=P[:, [1, 0]].astype(np.float32),
             names=np.array(["tumor", "normal"]), groups=np.array(["tumor", "normal"]),
             is_tumor=np.array([True, False]))
    E = np.vstack([P.T, rng.standard_normal((30, P.shape[0]))])
    np.savez(os.path.join(root, "bank.npz"), embeddings=(E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32))
    out = os.path.join(root, "out")
    _, args = parse(["--folder", os.path.join(root, "feat"), "--csv_path", os.path.join(root, "C.csv"),
                     "--concepts", os.path.join(root, "concepts.npz"), "--bank", os.path.join(root, "bank.npz"),
                     "--rank", "8", "--pool", "lse", "top1", "--lam_tail", "0", "1", "--extra_normals", "0", "8",
                     "--k_shots", "8", "--seeds", "0", "1", "--epochs", "120", "--lr", "1e-2",
                     "--reference_text_prototypes", os.path.join(root, "protos.npy"), "--ref_topk", "1",
                     "--xml_dir", os.path.join(root, "xml"), "--h5_dir", os.path.join(root, "h5"),
                     "--out_dir", out])
    produced, summ = run(args)
    assert len(produced) == 1 + 2 * 2 * 2
    evals = {p.rsplit("=", 1)[1]: pd.read_csv(p.rsplit("=", 1)[1]) for p in produced}
    sets = {frozenset(zip(d.seed, d.slide_id)) for d in evals.values()}
    assert len(sets) == 1, "configurations must share one eval set"
    for d in evals.values():
        assert (d.pred_label == np.where(d.slide_score > 0, "Tumor", "Normal")).all()
    tr = pd.read_csv(os.path.join(out, "ConceptTail_lse_r8_tail1_x8_training.csv"))
    assert (tr.n_support_normals == 16).all() and (tr.tail_J >= 1).all()
    pr = pd.read_csv(os.path.join(out, "proposition_by_bin.csv"))
    assert {"observed_mean", "pred_mean", "pred_simple_mean"} <= set(pr.columns)
    assert os.path.exists(os.path.join(out, "proposition_by_witness_rate.png"))
    print(summ.drop(columns="path", errors="ignore").to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"self_test: OK  (figure: {out}/proposition_by_witness_rate.png)")
    return out


def main():
    p, args = parse()
    if args.self_test:
        self_test()
        return
    if not args.folder or not args.concepts:
        p.error("--folder and --concepts are required (or --self_test)")
    produced, _ = run(args)
    if not args.no_compare:
        compare(args, produced)


if __name__ == "__main__":
    main()
