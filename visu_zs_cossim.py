"""
visu_zs_cossim.py

Ground truth vs RAW CONCH cosine similarity to each text prototype, for one CAMELYON16
slide: no logit_scale, no softmax, so the Tumor and Normal maps are independent and do
not sum to anything.

Draws   GT annotation | cos(patch, t_Tumor) | cos(patch, t_Normal)
with ONE shared colour scale for both similarity panels (default: the 1st-99th
percentile of both maps together, because cosine similarities of CONCH patches sit in a
narrow band, e.g. 0.1-0.4, and a [-1, 1] scale would show nothing). Set --vmin/--vmax to
fix the scale across slides. The header gives the GT label and the witness rate (tumor
area %, centre-rule %, lesions, largest lesion stage) from utils/witness_rate.py.

Patches are drawn at their true level-0 position and size; the annotation outline is
repeated on both similarity panels.

GPU recommended (CONCH projection head).

    python visu_zs_cossim.py --slide_id test_048 \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --wsi_dir $SCRATCH/camelyon16/images --xml_dir $SCRATCH/camelyon16/annotations
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
    p.add_argument("--csv_path", default=None)
    p.add_argument("--folder", default=None, help="Unused; kept for old command lines.")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--checkpoint_path",
                   default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--vmin", type=float, default=None)
    p.add_argument("--vmax", type=float, default=None)
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--thumb_max_dim", type=int, default=3000)
    p.add_argument("--mpp_level0", type=float, default=None)
    p.add_argument("--out_dir", default="./visualizations")
    p.add_argument("--exp_name", default="")
    args = p.parse_args()

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

    h5_path = os.path.join(args.h5_dir, f"{args.slide_id}.h5")
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:].astype(float)
        ps = read_patch_size_level0(f, coords)
        sim = Z.patch_cosine(f["features"][:], conch, protos, device)
    cos_t, cos_n = sim[:, CLASSES.index("Tumor")], sim[:, CLASSES.index("Normal")]

    both = np.concatenate([cos_t, cos_n])
    vmin = args.vmin if args.vmin is not None else float(np.percentile(both, 1))
    vmax = args.vmax if args.vmax is not None else float(np.percentile(both, 99))

    ann, xml_path = V.load_annotations(args.xml_dir, args.slide_id)
    stats = V.witness_summary(h5_path, xml_path, gt, args.mpp_level0 or DEFAULT_MPP_LEVEL0)
    thumb, dsx, dsy = V.load_thumbnail(V.find_slide_path(args.wsi_dir, args.slide_id),
                                       args.thumb_max_dim)

    fig, axes = V.make_figure(thumb, 3, panel_width_in=5.5, header_lines=3)
    V.draw_gt_panel(axes[0], thumb, ann, dsx, dsy, gt)
    V.draw_patch_map(axes[1], thumb, coords, ps, cos_t, dsx, dsy, "cos(patch, Tumor prompt)",
                     cmap=args.cmap, vmin=vmin, vmax=vmax, ann=ann)
    im = V.draw_patch_map(axes[2], thumb, coords, ps, cos_n, dsx, dsy,
                          "cos(patch, Normal prompt)", cmap=args.cmap, vmin=vmin, vmax=vmax,
                          ann=ann)
    V.add_colorbar(fig, axes[2], im, "raw cosine similarity (shared scale)")
    margin = cos_t - cos_n
    V.add_header(
        fig, f"{args.slide_id}  ·  GT {gt}  ·  raw CONCH cosine similarity",
        [V.witness_text(stats),
         f"cos Tumor [{cos_t.min():.3f}, {cos_t.max():.3f}]  ·  cos Normal "
         f"[{cos_n.min():.3f}, {cos_n.max():.3f}]  ·  patches with cos Tumor > cos Normal: "
         f"{int((margin > 0).sum())}/{len(margin)}  ·  prompts: "
         f"{os.path.basename(args.text_prototypes)}"],
        footer=f"Colour scale {vmin:.3f} to {vmax:.3f}"
               + ("" if args.vmin is not None else " (1st-99th percentile of both maps)")
               + f"   ·   patches at true level-0 size ({ps} px); white outline = annotation")
    tag = f"_{args.exp_name}" if args.exp_name else ""
    V.save(fig, os.path.join(args.out_dir, f"{args.slide_id}_gt_vs_cossim{tag}.png"))
    print(f"{args.slide_id}: GT {gt}; {V.witness_text(stats)}")


if __name__ == "__main__":
    main()
