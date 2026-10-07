import h5py, numpy as np, os
from glob import glob
from tqdm import tqdm

src = "/project/rrg-josedolz/natgill/data/camelyon16/trident/20x_512px_0px_overlap/features_conch_v1"
dst = "/project/rrg-josedolz/natgill/data/camelyon16_milformat/CAMELYON16/CONCH"  # -> folder/CAMELYON16/CONCH/<WSI>.npy
os.makedirs(dst, exist_ok=True)

for f in tqdm(glob(os.path.join(src, "*.h5"))):
    slide_id = os.path.splitext(os.path.basename(f))[0]
    with h5py.File(f, "r") as h5f:
        feats = h5f["features"][:]
    np.save(os.path.join(dst, f"{slide_id}.npy"), feats)