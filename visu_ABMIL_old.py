import os
import h5py
import numpy as np
import torch
import openslide
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from matplotlib.patches import Polygon as MplPolygon

import sys
sys.path.insert(0, "/project/rrg-josedolz/natgill/baselines/MIL-Adapter")  # so utils.* imports resolve
from utils.adapters import ZSMIL
from utils.trainer import train_model
from utils.utils import set_random_seeds, get_project_data, load_data, fewshot_sampling

# ---- config ----
folder = "/project/rrg-josedolz/natgill/data/camelyon16_milformat"
h5_dir = "/project/rrg-josedolz/natgill/data/camelyon16/trident/20x_512px_0px_overlap/features_conch_v1"
slide_dir = "/scratch/natgill/camelyon16/images"           # wherever the .tif originals live
xml_dir = "/scratch/natgill/camelyon16/annotations"  # adjust to your actual path
project, encoder, aggregator = "CAMELYON16", "CONCH", "ABMIL"
k_shots, seed, epochs, lr = 16, 0, 20, 1e-3               # pick the seed you want to inspect
target_wsi = "test_033"                                    # a tumor slide id from your validation set
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- 1. reproduce the exact training run for this seed ----
classes, classes_id = get_project_data(project)
X, Y, WSI = load_data(project=project, encoder=encoder, folder=folder, classes=classes)
set_random_seeds(seed_value=seed)
train_data, val_data, train_labels, val_labels = fewshot_sampling(X=X, Y=Y, k_shots=k_shots, seed=seed)

text_prototypes = np.load(f"./local_data/prompts/{encoder}/{project}.npy")
model = ZSMIL(text_embeddings=text_prototypes, aggregator=aggregator, init="random").to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = torch.nn.CrossEntropyLoss(reduction="sum")
train_model(model, optimizer, criterion, scheduler, train_data, train_labels, epochs)
model.eval()

# ---- 2. extract per-patch attention for the target slide ----
wsi_idx = list(WSI).index(target_wsi)
features = torch.tensor(X[wsi_idx], dtype=torch.float32).to(device)
with torch.no_grad():
    embedding, w = model.ABMIL(features)   # w: [N_patches, 1]
attention = w.squeeze(-1).cpu().numpy()
attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)  # normalize for display

# ---- 3. get patch coordinates + size from the ORIGINAL h5 (not the stripped .npy) ----
with h5py.File(os.path.join(h5_dir, f"{target_wsi}.h5"), "r") as f:
    coords = f["coords"][:]                                    # [N_patches, 2], level-0 (x, y)
    patch_size_level0 = int(f["coords"].attrs["patch_size_level0"])
assert len(coords) == len(attention), "coords/attention length mismatch — check patch order"

# ---- 4. build ground-truth tumor mask from the XML annotation, at thumbnail resolution ----
slide = openslide.OpenSlide(os.path.join(slide_dir, f"{target_wsi}.tif"))
thumb_level = slide.get_best_level_for_downsample(64)
thumb = slide.read_region((0, 0), thumb_level, slide.level_dimensions[thumb_level]).convert("RGB")
downsample = slide.level_downsamples[thumb_level]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax in axes:
    ax.imshow(thumb)
    ax.axis("off")

xml_path = os.path.join(xml_dir, f"{target_wsi}.xml")
if os.path.exists(xml_path):
    tree = ET.parse(xml_path)
    for annotation in tree.iter("Annotation"):
        pts = [(float(c.get("X")) / downsample, float(c.get("Y")) / downsample)
               for c in annotation.iter("Coordinate")]
        axes[0].add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor="lime", linewidth=1.5))
axes[0].set_title(f"{target_wsi} — tumor annotation (GT)")

# ---- 5. overlay attention heatmap, patch by patch ----
patch_size_thumb = patch_size_level0 / downsample
sc = axes[1].scatter(coords[:, 0] / downsample, coords[:, 1] / downsample,
                      c=attention, cmap="jet", s=(patch_size_thumb ** 2) / 20, alpha=0.5, marker="s")
plt.colorbar(sc, ax=axes[1], label="ABMIL attention (normalized)")
axes[1].set_title(f"{target_wsi} — ABMIL attention")

plt.tight_layout()
os.makedirs("attention_map", exist_ok=True)
plt.savefig(f"attention_map/attention_vs_gt_{target_wsi}_seed{seed}.png", dpi=200)
print("Saved.")