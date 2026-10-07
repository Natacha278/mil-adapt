"""
analyze_zeroshot_patch_accuracy.py

Measures CONCH zero-shot PATCH-LEVEL classification accuracy for CAMELYON16, reported
separately for Normal slides and Tumor slides (as two different, non-comparable
questions, since the two have different per-patch ground truth sources):

  - Normal slides: every patch is, by construction, ground-truth Normal (the slide has
    no tumor anywhere). Patch-level accuracy = fraction of patches the zero-shot model
    (argmax over per-patch cosine similarity to the "Normal"/"Tumor" text prototypes)
    predicts as Normal. Pooled across ALL patches of ALL normal slides.

  - Tumor slides: per-patch ground truth comes from the CAMELYON16 lesion-annotation
    XML (ASAP polygons) -- a patch's center is tested for point-in-polygon membership,
    same convention as visu_ABMIL.py's compute_tumor_patch_pct. A tumor slide's patches
    are a MIX of in-annotation (truly tumor) and out-of-annotation (truly normal, e.g.
    surrounding tissue) patches, so accuracy here is checked against that per-patch
    label, not "predict Tumor everywhere". Pooled across ALL patches of ALL tumor
    slides that have an XML annotation (tumor slides with a missing XML are skipped --
    reported separately, since there's no ground truth to check them against).

  Also reports, as a secondary breakdown (not the two headline numbers, but useful for
  interpreting them): within tumor slides only, accuracy split further into the
  in-annotation (true tumor) patches vs out-of-annotation (true normal) patches --
  this shows whether errors on tumor slides are mostly missed tumor regions, mostly
  false positives in surrounding normal tissue, or both.

Patch-level predicted class = argmax over raw cosine similarity to the text
prototypes (CONCH patch features -> forward_project() -> normalize -> dot product with
the already-normalized prototypes). argmax is invariant to logit_scale/softmax, so this
is identical to argmax of the softmax probabilities used in visu_zs.py -- no slide-level
MI-Zero top-j pooling is involved here, since this measures PER-PATCH accuracy, not the
slide-level prediction.

Place this file in the same directory as main.py so it can import directly from
utils.* like main.py does.

Run on a GPU allocation (salloc/sbatch), not the login node.
"""

import os
import argparse

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib.path import Path as MplPath
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained

# --- Direct imports from the MIL-Adapter repo (this script sits next to main.py) ---
from utils.utils import load_data
from visu_ABMIL import parse_camelyon16_xml


def get_project_data_camelyon16():
    classes = ["Normal", "Tumor"]
    classes_id = ["Normal", "Tumor"]
    return classes, classes_id


@torch.no_grad()
def compute_patch_predictions(h5_path, conch_model, device, text_prototypes, batch_size=4096):
    """Returns (pred_ids [N], coords [N,2], patch_size_level0 (int))."""
    with h5py.File(h5_path, "r") as h5f:
        raw_features = h5f["features"][:]
        coords = h5f["coords"][:]
        patch_size_level0 = h5f.attrs.get("patch_size_level0", h5f.attrs.get("patch_size", 512))

    feats = torch.tensor(raw_features, dtype=torch.float32, device=device)

    projected_chunks = []
    for i in range(0, feats.shape[0], batch_size):
        chunk = feats[i:i + batch_size]
        proj_chunk = conch_model.visual.forward_project(chunk)
        proj_chunk = torch.nn.functional.normalize(proj_chunk, dim=-1)
        projected_chunks.append(proj_chunk)
    projected = torch.cat(projected_chunks, dim=0)

    similarity = projected @ text_prototypes  # [N, n_classes]
    pred_ids = similarity.argmax(dim=1).cpu().numpy()

    return pred_ids, coords, int(patch_size_level0)


def patch_gt_from_xml(coords, patch_size_level0, xml_path, tumor_idx, normal_idx):
    """Per-patch ground-truth class id from point-in-polygon membership."""
    polygons = parse_camelyon16_xml(xml_path)
    n_patches = coords.shape[0]
    gt_ids = np.full(n_patches, normal_idx, dtype=int)
    if len(polygons) == 0:
        return gt_ids

    centers = coords + patch_size_level0 / 2.0
    inside = np.zeros(n_patches, dtype=bool)
    for poly in polygons:
        path = MplPath(poly)
        inside |= path.contains_points(centers)
    gt_ids[inside] = tumor_idx
    return gt_ids


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
    parser.add_argument("--out_csv", default="zeroshot_patch_accuracy_per_slide.csv")
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

    normal_correct, normal_total = 0, 0
    tumor_slide_correct, tumor_slide_total = 0, 0          # pooled, all patches in tumor slides
    tumor_region_correct, tumor_region_total = 0, 0        # in-annotation patches only
    surround_region_correct, surround_region_total = 0, 0  # out-of-annotation patches only
    n_tumor_slides_skipped_no_xml = 0

    per_slide_rows = []

    for i in tqdm(range(len(WSI)), desc="Zero-shot patch accuracy"):
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
            correct = int((pred_ids == normal_idx).sum())
            total = len(pred_ids)
            normal_correct += correct
            normal_total += total
            per_slide_rows.append({
                "WSI": slide_id, "gt_label": gt_label, "n_patches": total,
                "n_correct": correct, "patch_acc": correct / total if total else np.nan,
                "n_tumor_region_patches": 0, "n_tumor_region_correct": np.nan,
                "n_surround_region_patches": np.nan, "n_surround_region_correct": np.nan,
            })

        elif gt_label == "Tumor":
            xml_path = os.path.join(args.xml_dir, f"{slide_id}.xml")
            if not os.path.exists(xml_path):
                n_tumor_slides_skipped_no_xml += 1
                print(f"[skip, no GT] {slide_id}: Tumor slide with no annotation xml at {xml_path}")
                continue

            gt_ids = patch_gt_from_xml(coords, patch_size_level0, xml_path, tumor_idx, normal_idx)
            correct_mask = (pred_ids == gt_ids)

            total = len(pred_ids)
            correct = int(correct_mask.sum())
            tumor_slide_correct += correct
            tumor_slide_total += total

            in_mask = (gt_ids == tumor_idx)
            out_mask = (gt_ids == normal_idx)
            n_in, n_out = int(in_mask.sum()), int(out_mask.sum())
            c_in = int(correct_mask[in_mask].sum()) if n_in else 0
            c_out = int(correct_mask[out_mask].sum()) if n_out else 0
            tumor_region_correct += c_in
            tumor_region_total += n_in
            surround_region_correct += c_out
            surround_region_total += n_out

            per_slide_rows.append({
                "WSI": slide_id, "gt_label": gt_label, "n_patches": total,
                "n_correct": correct, "patch_acc": correct / total if total else np.nan,
                "n_tumor_region_patches": n_in,
                "n_tumor_region_correct": c_in / n_in if n_in else np.nan,
                "n_surround_region_patches": n_out,
                "n_surround_region_correct": c_out / n_out if n_out else np.nan,
            })

    df = pd.DataFrame(per_slide_rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-slide results -> {args.out_csv}")

    normal_acc = normal_correct / normal_total if normal_total else float("nan")
    tumor_acc = tumor_slide_correct / tumor_slide_total if tumor_slide_total else float("nan")
    tumor_region_acc = tumor_region_correct / tumor_region_total if tumor_region_total else float("nan")
    surround_region_acc = surround_region_correct / surround_region_total if surround_region_total else float("nan")

    print("\n" + "=" * 70)
    print("Zero-shot PATCH-LEVEL accuracy")
    print("=" * 70)
    print(f"Normal slides:  {normal_acc:.4f}  ({normal_correct}/{normal_total} patches, "
          f"{(df['gt_label'] == 'Normal').sum()} slides)")
    print(f"Tumor slides:   {tumor_acc:.4f}  ({tumor_slide_correct}/{tumor_slide_total} patches, "
          f"{(df['gt_label'] == 'Tumor').sum()} slides with annotation"
          + (f", {n_tumor_slides_skipped_no_xml} skipped -- no annotation xml" if n_tumor_slides_skipped_no_xml else "")
          + ")")
    print("-" * 70)
    print("Breakdown within tumor slides (not pooled above, shown for context):")
    print(f"  tumor-region patches only:     {tumor_region_acc:.4f}  "
          f"({tumor_region_correct}/{tumor_region_total})  <- tumor recall at patch level")
    print(f"  surrounding normal patches:    {surround_region_acc:.4f}  "
          f"({surround_region_correct}/{surround_region_total})")
    print("=" * 70)


if __name__ == "__main__":
    main()
