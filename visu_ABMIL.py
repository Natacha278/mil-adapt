"""
visu_ABMIL_zs.py

GT + trained-MIL-Adapter-only visualization for CAMELYON16. Stripped-down variant of
visu_ABMIL.py that drops the zero-shot CONCH computation entirely: no CONCH model
loading, no forward_project(), no MI-Zero top-j pooling. Place this file in the same
directory as main.py so it can import directly from utils.* like main.py does.

For a given slide, this:
  1. Reproduces a MIL-Adapter few-shot run using the ACTUAL repo classes
     (utils.adapters.{ZSMIL,TaskRes,CLIPAdapter,TIPAdapter}), aggregator=ABMIL,
     fixed seed -- training is a matter of seconds, so no checkpoint save/load needed.
  2. Extracts the trained ABMIL attention weights for the target slide and the
     chosen adapter's predicted label.
  3. Parses the CAMELYON16 ground-truth tumor annotation (ASAP XML) for that slide.
  4. Renders a 2-panel figure: GT overlay | trained attention, with GT / adapter
     predicted labels in the title.

Note: --text_prototypes is still required -- ZSMIL/TaskRes/CLIPAdapter/TIPAdapter all
initialize their classifier from the precomputed text embeddings, independent of the
zero-shot visualization that's been removed here.

Run on a GPU allocation (salloc/sbatch), not the login node.

Requirements (same venv used for MIL-Adapter + CONCH):
    pip install openslide-python h5py matplotlib
    module load openslide   (C library; openslide-python is the separate pip package)
"""

import os
import types
import argparse
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import openslide

# --- Direct imports from the MIL-Adapter repo (this script sits next to main.py) ---
from utils.adapters import ZSMIL, TaskRes, CLIPAdapter, TIPAdapter
from utils.trainer import train_model
from utils.utils import set_random_seeds, load_data, fewshot_sampling


ADAPTER_CLASSES = {
    "ZSMIL": ZSMIL,
    "TaskRes": TaskRes,
    "CLIPAdapter": CLIPAdapter,
    "TIPAdapter": TIPAdapter,
}

UNIMPLEMENTED_ADAPTERS = {"LR"}


def _clipadapter_forward_fixed(self, features):
    if self.aggregator == "BGAP":
        embedding = torch.mean(features, dim=0)
    elif self.aggregator == "BGMP":
        embedding = torch.max(features, dim=0)[0]
    elif self.aggregator == "ABMIL":
        embedding, _w = self.ABMIL(features)  # FIX: unpack (embedding, w)
    elif self.aggregator == "TransMIL":
        embedding = self.TransMIL(features)
    prototype = self.classifier
    embedding_res = self.adapter(embedding)
    embedding = self.ratio * embedding_res + (1 - self.ratio) * embedding
    embedding_norm = embedding / embedding.norm(dim=-1, keepdim=True)
    prototype_norm = prototype / prototype.norm(dim=0, keepdim=True)
    output = embedding_norm @ prototype_norm * self.logit_scale
    return output, embedding_norm


def _tipadapter_forward_fixed(self, features):
    if self.aggregator == "BGAP":
        embedding = torch.mean(features, dim=0)
    elif self.aggregator == "BGMP":
        embedding = torch.max(features, dim=0)[0]
    elif self.aggregator == "ABMIL":
        embedding, _w = self.ABMIL(features)  # FIX: unpack (embedding, w)
    elif self.aggregator == "TransMIL":
        embedding = self.TransMIL(features)
    prototype = self.classifier
    embedding_norm = embedding / embedding.norm(dim=-1, keepdim=True)
    prototype_norm = prototype / prototype.norm(dim=0, keepdim=True)
    clip_logits = embedding_norm @ prototype_norm * self.logit_scale

    cache_keys = self.cache_keys / self.cache_keys.norm(dim=-1, keepdim=True)
    affinity = embedding_norm @ cache_keys.t()
    affinity = torch.exp(((-1) * (self.beta - self.beta * affinity)))
    cache_logits = affinity @ self.cache_values
    output = clip_logits + cache_logits * self.alpha
    return output, embedding_norm


def patch_known_bugs(model, adapter_name, aggregator):
    if aggregator != "ABMIL":
        return model
    if adapter_name == "CLIPAdapter":
        model.forward = types.MethodType(_clipadapter_forward_fixed, model)
    elif adapter_name == "TIPAdapter":
        model.forward = types.MethodType(_tipadapter_forward_fixed, model)
    return model


def get_project_data_camelyon16():
    classes = ["Normal", "Tumor"]
    classes_id = ["Normal", "Tumor"]
    return classes, classes_id


def parse_camelyon16_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    polygons = []
    for annotation in root.iter("Annotation"):
        coords_elem = annotation.find("Coordinates")
        if coords_elem is None:
            continue
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in coords_elem.iter("Coordinate")]
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


# =====================================================================================
# Plotting
# =====================================================================================

def plot_two_panel(slide_path, xml_path, coords, trained_attention,
                    gt_label, adapter_name, pred_label_adapter, pred_conf_adapter,
                    out_path, thumb_max_dim=3000, panel_width_in=7.0, save_dpi=300):
    slide = openslide.OpenSlide(slide_path)
    level0_w, level0_h = slide.dimensions
    downsample = max(level0_w, level0_h) / thumb_max_dim
    thumb_w, thumb_h = int(level0_w / downsample), int(level0_h / downsample)
    thumb = slide.get_thumbnail((thumb_w, thumb_h))

    aspect = thumb_h / thumb_w
    panel_height_in = panel_width_in * aspect
    fig, axes = plt.subplots(1, 2, figsize=(panel_width_in * 2, panel_height_in + 1.0))
    fig.subplots_adjust(wspace=0.03, top=0.88)

    fig.suptitle(
        f"GT: {gt_label}    |    {adapter_name}+ABMIL pred: {pred_label_adapter} "
        f"({pred_conf_adapter:.2f})",
        fontsize=13
    )

    axes[0].imshow(thumb)
    axes[0].set_title("Ground truth (tumor annotation)")
    if xml_path and os.path.exists(xml_path):
        polygons = parse_camelyon16_xml(xml_path)
        patches_list = [mpatches.Polygon([(x / downsample, y / downsample) for x, y in poly], closed=True)
                         for poly in polygons]
        pc = PatchCollection(patches_list, facecolor="red", alpha=0.35, edgecolor="darkred")
        axes[0].add_collection(pc)
    else:
        axes[0].text(0.5, 0.5, "No tumor annotation\n(normal slide)",
                      transform=axes[0].transAxes, ha="center", va="center",
                      fontsize=12, color="gray")
    axes[0].axis("off")

    axes[1].imshow(thumb)
    axes[1].set_title(f"Trained few-shot {adapter_name}+ABMIL attention")
    scaled_coords = coords / downsample
    sc1 = axes[1].scatter(scaled_coords[:, 0], scaled_coords[:, 1],
                           c=trained_attention, cmap="jet", s=6, alpha=0.7, marker="s")
    plt.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].axis("off")

    plt.savefig(out_path, dpi=save_dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_id", required=True, help="e.g. tumor_026")
    parser.add_argument("--wsi_dir", required=True, help="Directory containing the WSI files (.tif/.svs).")
    parser.add_argument("--xml_dir", required=True, help="Directory containing lesion_annotations/*.xml.")
    parser.add_argument("--h5_dir",
                         default="/project/rrg-josedolz/natgill/data/camelyon16/trident/20x_512px_0px_overlap/features_conch_v1",
                         help="Original TRIDENT h5 files -- used only for patch coords "
                              "(to plot the attention scatter at the right locations).")
    parser.add_argument("--folder", required=True)
    parser.add_argument("--project", default="CAMELYON16")
    parser.add_argument("--encoder", default="CONCH")
    parser.add_argument("--adapter", default="ZSMIL",
                         choices=["ZSMIL", "TaskRes", "CLIPAdapter", "TIPAdapter"])
    parser.add_argument("--aggregator", default="ABMIL", choices=["ABMIL"])
    parser.add_argument("--init", choices=["ZS", "random"], default="ZS")
    parser.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy",
                         help="Still required: ZSMIL/TaskRes/CLIPAdapter/TIPAdapter all "
                              "initialize their classifier from these text embeddings.")
    parser.add_argument("--k_shots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="./visualizations")
    args = parser.parse_args()

    if args.adapter in UNIMPLEMENTED_ADAPTERS:
        raise ValueError(f"--adapter {args.adapter} is not implemented in MIL-Adapter's utils/adapters.py.")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU visible — utils/trainer.py hardcodes .cuda(), this will fail on CPU.")

    classes, classes_id = get_project_data_camelyon16()
    text_prototypes_np = np.load(args.text_prototypes)

    X, Y, WSI = load_data(folder=args.folder, project=args.project,
                           encoder=args.encoder, classes=classes)

    set_random_seeds(seed_value=args.seed)
    train_data, val_data, train_labels, val_labels = fewshot_sampling(
        X=X, Y=Y, k_shots=args.k_shots, seed=args.seed
    )

    adapter_cls = ADAPTER_CLASSES[args.adapter]
    if args.adapter == "ZSMIL":
        model = adapter_cls(text_embeddings=text_prototypes_np, aggregator=args.aggregator, init=args.init)
    elif args.adapter == "TaskRes":
        model = adapter_cls(text_embeddings=text_prototypes_np, aggregator=args.aggregator)
    elif args.adapter == "CLIPAdapter":
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

    slide_idx = np.where(WSI == args.slide_id)[0]
    if len(slide_idx) == 0:
        raise ValueError(f"Slide '{args.slide_id}' not found in {args.project}.csv / WSI list.")
    slide_idx = int(slide_idx[0])

    gt_label = classes_id[Y[slide_idx]]
    slide_features_np = X[slide_idx]

    in_train = any(f is X[slide_idx] for f in train_data)
    print(f"Slide {args.slide_id} was in {'TRAIN' if in_train else 'VAL'} split "
          f"for seed={args.seed}, k_shots={args.k_shots}")

    model.eval()
    with torch.no_grad():
        batch = torch.tensor(slide_features_np, dtype=torch.float32, device=device)
        logits = model(batch)[0]
        probs = logits.softmax(dim=0).cpu().numpy()
        pred_id_adapter = int(probs.argmax())
        _, w = model.ABMIL(batch)
        trained_attention = w.squeeze(-1).cpu().numpy()

    pred_label_adapter = classes_id[pred_id_adapter]
    pred_conf_adapter = float(probs[pred_id_adapter])

    h5_path = os.path.join(args.h5_dir, f"{args.slide_id}.h5")
    with h5py.File(h5_path, "r") as h5f:
        coords = h5f["coords"][:]

    if trained_attention.shape[0] != coords.shape[0]:
        raise ValueError(
            f"Patch count mismatch: milformat features have {trained_attention.shape[0]} patches, "
            f"h5 coords have {coords.shape[0]} for {args.slide_id}. "
            "Milformat .npy and original .h5 must come from the same extraction run."
        )

    slide_path_candidates = [
        os.path.join(args.wsi_dir, f"{args.slide_id}.tif"),
        os.path.join(args.wsi_dir, f"{args.slide_id}.svs"),
    ]
    slide_path = next((p for p in slide_path_candidates if os.path.exists(p)), None)
    if slide_path is None:
        raise FileNotFoundError(f"No WSI file found for {args.slide_id} in {args.wsi_dir}")

    xml_path = os.path.join(args.xml_dir, f"{args.slide_id}.xml")

    out_path = os.path.join(
        args.out_dir, f"{args.slide_id}_gt_{args.adapter}_{args.aggregator}_k{args.k_shots}.png"
    )
    plot_two_panel(
        slide_path=slide_path,
        xml_path=xml_path if os.path.exists(xml_path) else None,
        coords=coords,
        trained_attention=trained_attention,
        gt_label=gt_label,
        adapter_name=args.adapter,
        pred_label_adapter=pred_label_adapter,
        pred_conf_adapter=pred_conf_adapter,
        out_path=out_path,
    )


if __name__ == "__main__":
    main()
