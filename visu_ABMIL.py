"""
visu_ABMIL.py

Ground truth vs few-shot MIL-Adapter (ABMIL) attention, for one CAMELYON16 slide.

  1. Trains a MIL-Adapter exactly as main.py does (same split via fewshot_sampling, same
     optimiser, scheduler and loss), with the repo's own adapter classes.
  2. Scores the target slide: predicted label, P(Tumor), and the ABMIL attention weights.
  3. Draws   GT annotation | ABMIL attention   with a header giving the GT label, the
     prediction (and whether it is correct), the slide's split (TRAIN / VAL: a slide in
     the support set was seen during training), and the witness rate (tumor area %,
     centre-rule %, lesions and largest lesion stage) from utils/witness_rate.py.

Patches are drawn at their true level-0 position and size (utils/visualization.py);
the annotation outline is repeated on the attention panel so alignment can be judged.

ATTENTION SCALE. ABMIL weights sum to 1, so their raw values shrink with slide size.
--attn_scale relative (default) shows N * a_i: 1 = uniform attention, 10 = ten times
the uniform share, on a log colour scale. --attn_scale minmax rescales to [0, 1] per
slide (the old behaviour; not comparable across slides).

parse_camelyon16_xml is kept with its previous behaviour (flat list of all polygons,
including "_2" holes) because analyze_zeroshot_patch_accuracy.py and
analyze_zeroshot_patch_metrics.py import it from here. New code should use
utils.witness_rate.parse_camelyon16_xml.

GPU required (utils/trainer.py calls .cuda()).

    python visu_ABMIL.py --slide_id test_033 --adapter TaskRes --k_shots 16 --seed 0 \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
        --wsi_dir $SCRATCH/camelyon16/images --xml_dir $SCRATCH/camelyon16/annotations \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat
"""

import argparse
import os
import types
import xml.etree.ElementTree as ET

import numpy as np

CLASSES = ["Normal", "Tumor"]


# =====================================================================================
# Kept for backward compatibility (imported by two analysis scripts)
# =====================================================================================

def parse_camelyon16_xml(xml_path):
    """LEGACY: flat list of every annotation polygon, "_2" holes included."""
    polygons = []
    for annotation in ET.parse(xml_path).getroot().iter("Annotation"):
        ce = annotation.find("Coordinates")
        if ce is None:
            continue
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in ce.iter("Coordinate")]
        if len(pts) >= 3:
            polygons.append(pts)
    return polygons


def get_project_data_camelyon16():
    return list(CLASSES), list(CLASSES)


# =====================================================================================
# Adapter forward fixes (CLIPAdapter / TIPAdapter do not unpack ABMIL's (embedding, w))
# =====================================================================================

def _pool(self, features):
    import torch
    if self.aggregator == "BGAP":
        return torch.mean(features, dim=0)
    if self.aggregator == "BGMP":
        return torch.max(features, dim=0)[0]
    if self.aggregator == "ABMIL":
        return self.ABMIL(features)[0]
    return self.TransMIL(features)


def _clipadapter_forward_fixed(self, features):
    embedding = _pool(self, features)
    embedding = self.ratio * self.adapter(embedding) + (1 - self.ratio) * embedding
    e = embedding / embedding.norm(dim=-1, keepdim=True)
    p = self.classifier / self.classifier.norm(dim=0, keepdim=True)
    return e @ p * self.logit_scale, e


def _tipadapter_forward_fixed(self, features):
    import torch
    e = _pool(self, features)
    e = e / e.norm(dim=-1, keepdim=True)
    p = self.classifier / self.classifier.norm(dim=0, keepdim=True)
    clip_logits = e @ p * self.logit_scale
    keys = self.cache_keys / self.cache_keys.norm(dim=-1, keepdim=True)
    affinity = torch.exp(-(self.beta - self.beta * (e @ keys.t())))
    return clip_logits + (affinity @ self.cache_values) * self.alpha, e


def patch_known_bugs(model, adapter_name, aggregator):
    if aggregator == "ABMIL" and adapter_name == "CLIPAdapter":
        model.forward = types.MethodType(_clipadapter_forward_fixed, model)
    elif aggregator == "ABMIL" and adapter_name == "TIPAdapter":
        model.forward = types.MethodType(_tipadapter_forward_fixed, model)
    return model


def build_model(adapter, text_prototypes, aggregator, init, train_data, train_labels):
    from utils.adapters import ZSMIL, TaskRes, CLIPAdapter, TIPAdapter
    if adapter == "ZSMIL":
        return ZSMIL(text_embeddings=text_prototypes, aggregator=aggregator, init=init)
    if adapter == "TaskRes":
        return TaskRes(text_embeddings=text_prototypes, aggregator=aggregator)
    if adapter == "CLIPAdapter":
        return CLIPAdapter(text_embeddings=text_prototypes, aggregator=aggregator)
    return TIPAdapter(text_embeddings=text_prototypes, train_data=train_data,
                      train_labels=train_labels, aggregator=aggregator)


# =====================================================================================
# Main
# =====================================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--slide_id", required=True)
    p.add_argument("--wsi_dir", required=True)
    p.add_argument("--xml_dir", required=True)
    p.add_argument("--h5_dir",
                   default="/project/rrg-josedolz/natgill/data/camelyon16/trident/"
                           "20x_512px_0px_overlap/features_conch_v1",
                   help="TRIDENT h5 files: patch coordinates and patch size.")
    p.add_argument("--folder", required=True, help="MIL-format data root for load_data.")
    p.add_argument("--project", default="CAMELYON16")
    p.add_argument("--encoder", default="CONCH")
    p.add_argument("--adapter", default="TaskRes",
                   choices=["ZSMIL", "TaskRes", "CLIPAdapter", "TIPAdapter"])
    p.add_argument("--aggregator", default="ABMIL", choices=["ABMIL"])
    p.add_argument("--init", choices=["ZS", "random"], default="ZS",
                   help="ZSMIL only.")
    p.add_argument("--text_prototypes", default="local_data/prompts/CONCH/CAMELYON16.npy")
    p.add_argument("--k_shots", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--attn_scale", choices=["relative", "minmax"], default="relative")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--thumb_max_dim", type=int, default=3000)
    p.add_argument("--mpp_level0", type=float, default=None,
                   help="Microns per level-0 pixel for lesion sizes (default 0.243).")
    p.add_argument("--out_dir", default="./visualizations")
    p.add_argument("--exp_name", default="")
    args = p.parse_args()

    import torch
    from matplotlib.colors import LogNorm
    from utils.trainer import train_model
    from utils.utils import set_random_seeds, load_data, fewshot_sampling
    from utils.witness_rate import load_patch_grid, DEFAULT_MPP_LEVEL0
    from utils import visualization as V

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU visible; utils/trainer.py calls .cuda() and will fail.")

    X, Y, WSI = load_data(folder=args.folder, project=args.project,
                          encoder=args.encoder, classes=CLASSES)
    idx = np.where(WSI == args.slide_id)[0]
    if len(idx) == 0:
        raise SystemExit(f"{args.slide_id} not in local_data/csv/{args.project}.csv")
    idx = int(idx[0])
    gt = CLASSES[Y[idx]]

    # --- train exactly like main.py ------------------------------------------------
    set_random_seeds(seed_value=args.seed)
    train_data, _val_data, train_labels, _val_labels = fewshot_sampling(
        X=X, Y=Y, k_shots=args.k_shots, seed=args.seed)
    in_train = any(f is X[idx] for f in train_data)
    text_prototypes = np.load(args.text_prototypes)
    model = build_model(args.adapter, text_prototypes, args.aggregator, args.init,
                        train_data, train_labels).to(device)
    model = patch_known_bugs(model, args.adapter, args.aggregator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                  weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    train_model(model, optimizer, criterion, scheduler, train_data, train_labels, args.epochs)

    # --- score the slide ------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        bag = torch.tensor(X[idx], dtype=torch.float32, device=device)
        probs = model(bag)[0].softmax(dim=0).cpu().numpy()
        attention = model.ABMIL(bag)[1].squeeze(-1).cpu().numpy()
    pred = CLASSES[int(probs.argmax())]
    p_tumor = float(probs[CLASSES.index("Tumor")])

    # --- geometry, annotation, witness rate --------------------------------------------
    h5_path = os.path.join(args.h5_dir, f"{args.slide_id}.h5")
    grid = load_patch_grid(h5_path)
    if len(grid) != len(attention):
        raise SystemExit(f"{len(attention)} patches in the MIL features but {len(grid)} in "
                         f"{h5_path}: they must come from the same extraction.")
    ann, xml_path = V.load_annotations(args.xml_dir, args.slide_id)
    stats = V.witness_summary(h5_path, xml_path, gt, args.mpp_level0 or DEFAULT_MPP_LEVEL0)
    thumb, dsx, dsy = V.load_thumbnail(V.find_slide_path(args.wsi_dir, args.slide_id),
                                       args.thumb_max_dim)

    if args.attn_scale == "relative":
        vals = attention * len(attention)
        lo = max(float(np.percentile(vals, 1)), 1e-3)
        norm = LogNorm(vmin=lo, vmax=max(float(vals.max()), lo * 10))
        cbar_label = "attention × N  (1 = uniform, log scale)"
        kw = {"norm": norm}
    else:
        rng = attention.max() - attention.min()
        vals = (attention - attention.min()) / (rng if rng > 0 else 1.0)
        cbar_label = "attention, min-max scaled per slide"
        kw = {"vmin": 0, "vmax": 1}

    # --- figure ---------------------------------------------------------------------
    fig, axes = V.make_figure(thumb, 2, header_lines=3)
    V.draw_gt_panel(axes[0], thumb, ann, dsx, dsy, gt)
    im = V.draw_patch_map(axes[1], thumb, grid.coords, grid.patch_size, vals, dsx, dsy,
                          f"{args.adapter}+ABMIL attention", cmap=args.cmap, ann=ann, **kw)
    V.add_colorbar(fig, axes[1], im, cbar_label)

    split = ("TRAIN (support slide: seen during training)" if in_train else "VAL (held out)")
    V.add_header(
        fig,
        f"{args.slide_id}  ·  GT {gt}  ·  {args.adapter}+ABMIL {V.pred_phrase(gt, pred, p_tumor)}",
        [V.witness_text(stats),
         f"k = {args.k_shots} shots, seed {args.seed}, {args.epochs} epochs  ·  slide split: "
         f"{split}  ·  prompts: {os.path.basename(args.text_prototypes)}"],
        footer=f"Patches drawn at true level-0 size ({grid.patch_size} px). White outline: "
               "annotated tumor; dashed: annotated non-tumor holes.")
    tag = f"_{args.exp_name}" if args.exp_name else ""
    out = os.path.join(args.out_dir, f"{args.slide_id}_gt_vs_{args.adapter}_ABMIL_"
                                     f"k{args.k_shots}_seed{args.seed}{tag}.png")
    V.save(fig, out)
    print(f"{args.slide_id}: GT {gt}, pred {pred} (P(Tumor) {p_tumor:.3f}), "
          f"{'TRAIN' if in_train else 'VAL'} split; {V.witness_text(stats)}")


if __name__ == "__main__":
    main()
