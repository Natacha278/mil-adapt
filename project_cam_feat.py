"""
Replace the raw/unprojected CAMELYON16 CONCH patch embeddings currently at
    /project/rrg-josedolz/natgill/data/camelyon16_milformat/CAMELYON16/CONCH/*.npy
with embeddings passed through CONCH's projection head (model.visual.forward_project),
sourced fresh from the original TRIDENT h5 files so no double-processing occurs.

Run on a GPU allocation (salloc/sbatch), not the login node — this loads the CONCH model.
"""

import os
import glob
import argparse

import h5py
import numpy as np
import torch
from tqdm import tqdm

from conch.open_clip_custom import create_model_from_pretrained


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_h5_dir",
        default="/project/rrg-josedolz/natgill/data/camelyon16/trident/20x_512px_0px_overlap/features_conch_v1",
        help="Directory of original TRIDENT h5 feature files (untouched raw CONCH features).",
    )
    parser.add_argument(
        "--dst_npy_dir",
        default="/project/rrg-josedolz/natgill/data/camelyon16_milformat/CAMELYON16/CONCH",
        help="MIL-Adapter-format directory to overwrite with projected embeddings.",
    )
    parser.add_argument(
        "--checkpoint_path",
        default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="L2-normalize each patch embedding after projection (default: on). "
             "Pass --no-normalize to keep raw projected (unnormalized) vectors.",
    )
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument(
        "--backup_dir",
        default=None,
        help="Optional directory to copy the original (raw) .npy files into before overwriting. "
             "Recommended on first run.",
    )
    parser.add_argument("--batch_size", type=int, default=4096,
                         help="Patches per forward_project() call, to bound GPU memory.")
    args = parser.parse_args()

    os.makedirs(args.dst_npy_dir, exist_ok=True)
    if args.backup_dir:
        os.makedirs(args.backup_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU visible, this will be very slow. Are you on a login node?")

    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.checkpoint_path)
    model = model.to(device).eval()

    h5_files = sorted(glob.glob(os.path.join(args.src_h5_dir, "*.h5")))
    print(f"Found {len(h5_files)} h5 files in {args.src_h5_dir}")

    n_ok, n_skipped, n_failed = 0, 0, 0

    for h5_path in tqdm(h5_files):
        slide_id = os.path.splitext(os.path.basename(h5_path))[0]
        dst_path = os.path.join(args.dst_npy_dir, f"{slide_id}.npy")

        try:
            with h5py.File(h5_path, "r") as h5f:
                raw_features = h5f["features"][:]

            if args.backup_dir and os.path.exists(dst_path):
                backup_path = os.path.join(args.backup_dir, f"{slide_id}.npy")
                if not os.path.exists(backup_path):
                    np.save(backup_path, np.load(dst_path))

            raw_features = torch.tensor(raw_features, dtype=torch.float32, device=device)

            projected_chunks = []
            with torch.no_grad():
                for i in range(0, raw_features.shape[0], args.batch_size):
                    chunk = raw_features[i:i + args.batch_size]
                    proj_chunk = model.visual.forward_project(chunk)
                    if args.normalize:
                        proj_chunk = torch.nn.functional.normalize(proj_chunk, dim=-1)
                    projected_chunks.append(proj_chunk.cpu())

            projected = torch.cat(projected_chunks, dim=0).numpy().astype(np.float32)

            assert projected.shape[0] == raw_features.shape[0], "patch count mismatch after projection"

            np.save(dst_path, projected)
            n_ok += 1

        except Exception as e:
            print(f"[FAILED] {slide_id}: {e}")
            n_failed += 1

    print(f"\nDone. ok={n_ok} failed={n_failed} skipped={n_skipped} total={len(h5_files)}")
    print(f"Projected embeddings written to: {args.dst_npy_dir}")
    if args.backup_dir:
        print(f"Original raw embeddings backed up to: {args.backup_dir}")


if __name__ == "__main__":
    main()