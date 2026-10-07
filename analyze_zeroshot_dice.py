"""
analyze_zeroshot_dice.py

Measures CONCH zero-shot patch-level tumor-segmentation Dice score for CAMELYON16.

Dice is a region-overlap metric, so it's only meaningful for slides that actually have
a tumor region to overlap with -- it's computed per TUMOR slide (ones with a lesion-
annotation XML), not for Normal slides (whose ground-truth tumor mask is empty
everywhere, which would make Dice trivially 1 or 0 and not tell you anything about
segmentation quality). Normal-slide false-positive behavior is reported separately as a
sanity stat, not folded into the Dice numbers.

Per tumor slide:
    predicted tumor mask = argmax(cosine similarity to text prototypes) == "Tumor"
    GT tumor mask        = patch center inside a lesion-annotation XML polygon
                            (same convention as visu_ABMIL.py's compute_tumor_patch_pct)
    Dice = 2 * |pred ∩ gt| / (|pred| + |gt|)
         = 2*TP / (2*TP + FP + FN)

Reports per-slide Dice (saved to CSV) plus the mean and median Dice across tumor
slides -- the standard way CAMELYON-style tumor segmentation is summarized, since a
simple micro-average would let the few largest tumors dominate the score.

Reuses compute_patch_predictions / patch_gt_from_xml / get_project_data_camelyon16
from analyze_zeroshot_patch_accuracy.py rather than redefining them -- keep that file
in the same directory as this one and as main.py.

Run on a GPU allocation (salloc/sbatch), not the login node.
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained

# --- Direct imports from the MIL-Adapter repo (this script sits next to main.py) ---
from utils.utils import load_data
from analyze_zeroshot_patch_accuracy import (
    get_project_data_camelyon16,
    compute_patch_predictions,
    patch_gt_from_xml,
)


def dice_score(pred_mask, gt_mask):
    """
    Dice = 2*TP / (2*TP + FP + FN) = 2*|pred ∩ gt| / (|pred| + |gt|).

    Edge case: if the GT mask is empty (shouldn't normally happen for an XML-annotated
    tumor slide, but can if the polygon parse yields nothing), Dice is defined as 1.0
    when the prediction is also empty (correct: no tumor found) and 0.0 otherwise
    (false positives with nothing to overlap).
    """
    pred_sum = int(pred_mask.sum())
    gt_sum = int(gt_mask.sum())
    if gt_sum == 0:
        return 1.0 if pred_sum == 0 else 0.0
    intersection = int((pred_mask & gt_mask).sum())
    return 2.0 * intersection / (pred_sum + gt_sum)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_dir",
                         default="/project/rrg-josedolz/natgill/data/camelyon16/trident/20x_512px_0px_overlap/features_conch_v1")
    parser.add_argument("--xml_dir", required=True, help="Directory containing lesion_annotations/*.xml.")
    parser.add_argument("--folder", required=True,
                         help="Passed to utils.utils.load_data -- used to get the slide "
                              "list and slide-level GT labels (WSI, Y).")
    parser.add_argument("--project", default="CAMELYON16")
    parser.add_argument("--encoder", default="CONCH")
    parser.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    parser.add_argument("--checkpoint_path", default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--out_csv", default="zeroshot_dice_per_slide.csv")
    parser.add_argument("--split", choices=["all", "test", "train"], default="all",
                         help="CAMELYON16's OFFICIAL train/test split (independent of "
                              "MIL-Adapter's own few-shot train/val sampling, which this "
                              "zero-shot script never uses). Based on WSI naming "
                              "convention: official test slides are named "
                              "'{test_prefix}###' (e.g. test_001), everything else "
                              "('normal_###', 'tumor_###') is the training pool. "
                              "'all' (default) evaluates every slide, same as before.")
    parser.add_argument("--test_prefix", default="test_",
                         help="WSI filename prefix identifying the official CAMELYON16 "
                              "test slides, used only when --split is 'test' or 'train'.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    classes, classes_id = get_project_data_camelyon16()
    tumor_idx = classes_id.index("Tumor")
    normal_idx = classes_id.index("Normal")

    text_prototypes_np = np.load(args.text_prototypes)
    text_prototypes = torch.tensor(text_prototypes_np, dtype=torch.float32, device=device)

    conch_model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.checkpoint_path)
    conch_model = conch_model.to(device).eval()

    _X, Y, WSI = load_data(folder=args.folder, project=args.project,
                            encoder=args.encoder, classes=classes)

    if args.split != "all":
        is_test = np.array([str(w).startswith(args.test_prefix) for w in WSI])
        keep_mask = is_test if args.split == "test" else ~is_test
        n_before = len(WSI)
        WSI = WSI[keep_mask]
        Y = Y[keep_mask]
        print(f"--split {args.split}: kept {len(WSI)}/{n_before} slides "
              f"(prefix '{args.test_prefix}' = test)")

    dice_rows = []
    n_tumor_slides_skipped_no_xml = 0

    # Normal-slide sanity stat: fraction of patches falsely predicted Tumor.
    normal_fp_patches, normal_total_patches = 0, 0

    for i in tqdm(range(len(WSI)), desc="Zero-shot Dice"):
        slide_id = WSI[i]
        gt_label = classes_id[Y[i]]
        h5_path = os.path.join(args.h5_dir, f"{slide_id}.h5")
        if not os.path.exists(h5_path):
            print(f"[skip] {slide_id}: no h5 at {h5_path}")
            continue

        pred_ids, coords, patch_size_level0 = compute_patch_predictions(
            h5_path, conch_model, device, text_prototypes, batch_size=args.batch_size
        )

        if gt_label == "Normal":
            normal_fp_patches += int((pred_ids == tumor_idx).sum())
            normal_total_patches += len(pred_ids)
            continue

        if gt_label != "Tumor":
            continue

        xml_path = os.path.join(args.xml_dir, f"{slide_id}.xml")
        if not os.path.exists(xml_path):
            n_tumor_slides_skipped_no_xml += 1
            print(f"[skip, no GT] {slide_id}: Tumor slide with no annotation xml at {xml_path}")
            continue

        gt_ids = patch_gt_from_xml(coords, patch_size_level0, xml_path, tumor_idx, normal_idx)
        pred_mask = (pred_ids == tumor_idx)
        gt_mask = (gt_ids == tumor_idx)

        dice = dice_score(pred_mask, gt_mask)
        tp = int((pred_mask & gt_mask).sum())
        fp = int((pred_mask & ~gt_mask).sum())
        fn = int((~pred_mask & gt_mask).sum())
        tn = int((~pred_mask & ~gt_mask).sum())

        dice_rows.append({
            "WSI": slide_id, "n_patches": len(pred_ids),
            "n_gt_tumor_patches": int(gt_mask.sum()), "n_pred_tumor_patches": int(pred_mask.sum()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "dice": dice,
        })

    df = pd.DataFrame(dice_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-slide Dice -> {args.out_csv}")

    print("\n" + "=" * 70)
    print("Zero-shot patch-level tumor-segmentation Dice (tumor slides only)")
    print("=" * 70)
    if len(df) > 0:
        print(f"Slides evaluated: {len(df)}"
              + (f"  ({n_tumor_slides_skipped_no_xml} skipped -- no annotation xml)"
                 if n_tumor_slides_skipped_no_xml else ""))
        print(f"Mean Dice:    {df['dice'].mean():.4f}")
        print(f"Median Dice:  {df['dice'].median():.4f}")
        print(f"Std Dice:     {df['dice'].std():.4f}")
        print(f"Min / Max:    {df['dice'].min():.4f} / {df['dice'].max():.4f}")
    else:
        print("No tumor slides with annotations were evaluated.")
    print("-" * 70)
    if normal_total_patches:
        fp_rate = normal_fp_patches / normal_total_patches
        print(f"Normal-slide sanity check (not a Dice score): "
              f"{fp_rate:.4f} of patches ({normal_fp_patches}/{normal_total_patches}) "
              f"falsely predicted Tumor.")
    print("=" * 70)


if __name__ == "__main__":
    main()
