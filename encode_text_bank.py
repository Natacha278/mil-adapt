"""
encode_text_bank.py

Encode a bank of pathology descriptions with CONCH (one embedding per sentence) for the
text subspace B of ConceptTail MIL (utils/concept_tail_mil.py, --bank). B is the top-r
left singular vectors of the [D, n] embedding matrix, computed at model build time, so
one bank file serves every rank r.

Input JSON: a list of sentences, or {group: [sentences]} (keys starting with "_" ignored).
Output .npz: embeddings [n, D] float32 (unit rows), texts [n], groups [n].
Prints the singular-value spectrum (share of energy in the top r directions).

    python encode_text_bank.py \\
        --bank_json local_data/prompts/templates/pathology_text_bank.json \\
        --out local_data/prompts/CONCH/pathology_text_bank.npz
    python encode_text_bank.py --self_test
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_bank(path):
    with open(path) as f:
        spec = json.load(f)
    if isinstance(spec, list):
        spec = {"bank": spec}
    texts, groups = [], []
    for g, lst in spec.items():
        if g.startswith("_"):
            continue
        for s in lst:
            if isinstance(s, str) and s.strip():
                texts.append(s.strip())
                groups.append(g)
    if len(texts) < 2:
        raise ValueError(f"{path}: need at least 2 sentences")
    return texts, groups


def spectrum_report(E, ranks=(4, 8, 16, 32, 64)):
    s = np.linalg.svd(np.asarray(E, np.float64).T, compute_uv=False)
    e = np.cumsum(s ** 2) / np.sum(s ** 2)
    print("share of bank energy in the top-r text directions: "
          + ", ".join(f"r={r}: {e[min(r, len(e)) - 1]:.3f}" for r in ranks if r <= len(e)))
    return e


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank_json")
    p.add_argument("--out")
    p.add_argument("--checkpoint_path", default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--self_test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        import tempfile
        tmp = tempfile.mkdtemp()
        jp = os.path.join(tmp, "b.json")
        json.dump({"_comment": "x", "a": ["one", "two"], "b": ["three", " "]}, open(jp, "w"))
        texts, groups = load_bank(jp)
        assert texts == ["one", "two", "three"] and groups == ["a", "a", "b"]
        E = np.random.default_rng(0).standard_normal((30, 8))
        e = spectrum_report(E / np.linalg.norm(E, axis=1, keepdims=True))
        assert abs(e[-1] - 1) < 1e-9
        texts, _ = load_bank(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "local_data/prompts/templates/pathology_text_bank.json"))
        print(f"  repository bank: {len(texts)} sentences")
        print("self_test: OK")
        return
    if not args.bank_json or not args.out:
        p.error("--bank_json and --out are required (or --self_test)")
    import torch
    import torch.nn.functional as F
    from conch.open_clip_custom import get_tokenizer, tokenize
    from utils import zeroshot as Z
    texts, groups = load_bank(args.bank_json)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conch = Z.load_conch(args.checkpoint_path, device)
    tok = get_tokenizer()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch):
            out.append(F.normalize(conch.encode_text(tokenize(tok, texts[i:i + args.batch]).to(device)),
                                   dim=-1).cpu().numpy())
    E = np.concatenate(out).astype(np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, embeddings=E, texts=np.array(texts), groups=np.array(groups))
    spectrum_report(E)
    print(f"-> {args.out}: {len(texts)} sentences, D={E.shape[1]}")


if __name__ == "__main__":
    main()
