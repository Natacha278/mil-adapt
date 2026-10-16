"""
visu_zs.py

Ground truth vs CONCH zero-shot prediction, for one CAMELYON16 slide.

  Per patch:  P(Tumor) = softmax(logit_scale * [cos(p, t_Normal), cos(p, t_Tumor)]).
  Per slide:  MI-Zero top-j pooling of the raw similarities (each class's j highest
              patch similarities are averaged; argmax gives the label, and the
              softmax of logit_scale * pooled gives the displayed confidence).

Draws   GT annotation | per-patch P(Tumor)   with a header giving the GT label, the
zero-shot prediction at --slide_topj (and whether it is correct), and the witness rate
(tumor area %, centre-rule %, lesions, largest lesion stage) from utils/witness_rate.py.
The footer lists the prediction for every j in --topj.

Patches are drawn at their true level-0 position and size; the annotation outline is
repeated on the score panel.

GPU recommended (CONCH projection head).

    python visu_zs.py --slide_id test_033 \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --wsi_dir $SCRATCH/camelyon16/images --xml_dir $SCRATCH/camelyon16/annotations \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat
"""

import argparse
import os

import numpy as np

CLASSES = ["Normal", "Tumor"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide_id", required=True)
    p.add_argument("--wsi_dir", required=True)
    p.add_argument("--xml_dir", required=True)
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1")
    p.add_argument("--csv_path", default=None,
                   help="Project CSV for the GT label (default ./local_data/csv/<project>.csv).")
    p.add_argument("--folder", default=None, help="Unused; kept for old command lines.")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--checkpoint_path",
                   default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--topj", type=int, nargs="+", default=[1, 5, 10, 50, 100])
    p.add_argument("--slide_topj", type=int, default=50)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--thumb_max_dim", type=int, default=3000)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--out_dir", default="./visualizations")
    p.add_argument("--exp_name", default="")
    args = p.parse_args()
    if args.slide_topj not in args.topj:
        args.topj = sorted(set(args.topj) | {args.slide_topj})

    import h5py
    import pandas as pd
    import torch
    from utils import zeroshot as Z
    from utils import visualization as V
    from utils.witness_rate import read_patch_size_level0, DEFAULT_MPP_LEVEL0

    meta = pd.read_csv(args.csv_path or f"./local_data/csv/{args.project}.csv")
    row = meta[meta["WSI"] == args.slide_id]
    if row.empty:
        raise SystemExit(f"{args.slide_id} not in the project CSV")
    gt = row["GT"].iloc[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conch = Z.load_conch(args.checkpoint_path, device)
    protos = Z.load_prototypes(args.text_prototypes, device)
    scale = Z.logit_scale(conch)

    h5_path = os.path.join(args.h5_dir, f"{args.slide_id}.h5")
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:].astype(float)
        ps = read_patch_size_level0(f, coords)
        sim = Z.patch_cosine(f["features"][:], conch, protos, device)

    p_patch = Z.softmax(scale * sim, axis=1)[:, CLASSES.index("Tumor")]
    pooled, preds = Z.topj_pooling(sim, args.topj)
    pred = CLASSES[preds[args.slide_topj]]
    p_tumor = float(Z.softmax(scale * pooled[args.slide_topj])[CLASSES.index("Tumor")])

    ann, xml_path = V.load_annotations(args.xml_dir, args.slide_id)
    stats = V.witness_summary(h5_path, xml_path, gt, args.mpp_level0 or DEFAULT_MPP_LEVEL0)
    thumb, dsx, dsy = V.load_thumbnail(V.find_slide_path(args.wsi_dir, args.slide_id),
                                       args.thumb_max_dim)

    fig, axes = V.make_figure(thumb, 2, header_lines=3)
    V.draw_gt_panel(axes[0], thumb, ann, dsx, dsy, gt)
    im = V.draw_patch_map(axes[1], thumb, coords, ps, p_patch, dsx, dsy,
                          "Zero-shot per-patch P(Tumor)", cmap=args.cmap, vmin=0, vmax=1,
                          ann=ann)
    V.add_colorbar(fig, axes[1], im, "P(Tumor) per patch")
    n_pos = int((p_patch > 0.5).sum())
    V.add_header(
        fig,
        f"{args.slide_id}  ·  GT {gt}  ·  zero-shot (MI-Zero top-{args.slide_topj}) "
        f"{V.pred_phrase(gt, pred, p_tumor)}",
        [V.witness_text(stats),
         f"patches with P(Tumor) > 0.5: {n_pos}/{len(p_patch)} ({100 * n_pos / len(p_patch):.2f}%)"
         f"  ·  prompts: {os.path.basename(args.text_prototypes)}"],
        footer="MI-Zero prediction by top-j:  " + "   ".join(
            f"j={j}: {CLASSES[preds[j]]}" for j in args.topj)
               + f"   ·   patches at true level-0 size ({ps} px); white outline = annotation")
    tag = f"_{args.exp_name}" if args.exp_name else ""
    V.save(fig, os.path.join(args.out_dir, f"{args.slide_id}_gt_vs_zs{tag}.png"))
    print(f"{args.slide_id}: GT {gt}; " + ", ".join(
        f"top-{j} {CLASSES[preds[j]]}" for j in args.topj) + f"; {V.witness_text(stats)}")


if __name__ == "__main__":
    main()
