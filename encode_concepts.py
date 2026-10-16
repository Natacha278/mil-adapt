"""
encode_concepts.py

Encode a class JSON (the format of visu_zs_classes.py, WITH "groups") into the concept
file used by the multi-concept Text-TopK model (utils/text_topk_mil_torch.py,
MultiConceptTextTopKMIL; runner / main.py / analysis scripts: --concepts <file>).

Each class is one concept; its phrasings are encoded with CONCH, L2-normalised,
averaged and re-normalised (same as create_CONCH_prompt_feat.py). The tumor side is every
group whose name contains "tumor"/"tumour"/"cancer"; every other group (normal,
background, ...) is the normal side.

Output .npz: protos [D, C] float32 (unit columns), names [C], groups [C], is_tumor [C].
Prints the concept-concept cosine similarities (close tumor / normal pairs = concepts
CONCH's text space barely separates).

    python encode_concepts.py \
        --class_json local_data/prompts/templates_attribs/cam_attrib_normal_v2.json \
        --out local_data/prompts/CONCH/CAMELYON16_concepts_v2.npz
    python encode_concepts.py --self_test
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visu_zs_classes import _family, encode_classes, load_class_spec   # noqa: E402


def concept_arrays(names, groups, protos):
    if groups is None:
        raise ValueError("the class JSON needs 'groups' (e.g. {'normal': [...], 'tumor': [...]})")
    is_tumor = np.array([_family(g) == "tumor" for g in groups], bool)
    if is_tumor.all() or (~is_tumor).all():
        raise ValueError(f"need at least one tumor and one non-tumor concept (groups: {sorted(set(groups))})")
    P = np.asarray(protos, np.float32)
    P = P / np.linalg.norm(P, axis=0, keepdims=True)
    return {"protos": P, "names": np.array(names), "groups": np.array(groups), "is_tumor": is_tumor}


def report(arrs):
    import pandas as pd
    names = [str(n) for n in arrs["names"]]
    side = ["T" if t else "N" for t in arrs["is_tumor"]]
    lab = [f"{s}:{n}" for s, n in zip(side, names)]
    cos = arrs["protos"].T @ arrs["protos"]
    pd.set_option("display.width", 250)
    print("concept cosine similarities (T = tumor side, N = normal side):")
    print(pd.DataFrame(cos, index=lab, columns=lab).to_string(float_format=lambda v: f"{v:.2f}"))
    t, n = arrs["is_tumor"], ~arrs["is_tumor"]
    cross = cos[np.ix_(t, n)]
    i, j = np.unravel_index(np.argmax(cross), cross.shape)
    print(f"closest tumor/normal pair: {np.array(names)[t][i]} ~ {np.array(names)[n][j]} "
          f"(cos {cross[i, j]:.2f})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--class_json")
    p.add_argument("--out")
    p.add_argument("--checkpoint_path",
                   default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        import json
        import tempfile
        tmp = tempfile.mkdtemp()
        spec = {"classes": {"lymphocytes": "a", "germinal_center": ["b", "c"], "solid_nests": "d"},
                "groups": {"normal": ["lymphocytes", "germinal_center"], "tumor": ["solid_nests"]}}
        jp = os.path.join(tmp, "c.json")
        json.dump(spec, open(jp, "w"))
        names, texts, _, groups = load_class_spec(jp)
        arrs = concept_arrays(names, groups, np.random.default_rng(0).standard_normal((8, 3)))
        out = os.path.join(tmp, "concepts.npz")
        np.savez(out, **arrs)
        from utils.text_topk_mil_torch import load_concepts, make_text_topk
        P, n2, t2 = load_concepts(out)
        assert n2 == names and t2.tolist() == [False, False, True] and np.allclose(np.linalg.norm(P, axis=0), 1)
        m = make_text_topk(concepts=out)
        assert m.names == names and int(m.tumor_mask.sum()) == 1
        report(arrs)
        try:
            concept_arrays(names, None, P)
            raise AssertionError("missing groups accepted")
        except ValueError:
            pass
        print("self_test: OK")
        return
    if not args.class_json or not args.out:
        p.error("--class_json and --out are required (or --self_test)")
    import torch
    from utils import zeroshot as Z
    names, texts, _, groups = load_class_spec(args.class_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conch = Z.load_conch(args.checkpoint_path, device)
    arrs = concept_arrays(names, groups, encode_classes(conch, texts, device))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **arrs)
    report(arrs)
    print(f"-> {args.out}: {len(names)} concepts ({int(arrs['is_tumor'].sum())} tumor)")


if __name__ == "__main__":
    main()
