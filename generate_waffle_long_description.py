"""
Waffles a LONG, LLM-generated single-sentence description (the format used in
camelyon_description.json: one full descriptive sentence per class, fed to CONCH's
zero_shot_classifier with templates=["CLASSNAME"], i.e. used verbatim, unwrapped) by
replacing its content-bearing words with random words, while preserving:
  - the leading classname phrase (e.g. "Breast tumor epithelial tissue"),
  - the sentence's grammatical scaffold (articles, prepositions, conjunctions, and a
    configurable set of connective/structural words),
  - punctuation and, per WaffleCLIP's convention, roughly the original word lengths,

so the result has the SAME length, structure, and grammatical complexity as the real
description, but none of its histological content. This isolates the effect of the
specific descriptive content from the mere effect of having a longer, more elaborate
prompt (vs. a short classname) -- complementary to generate_waffle_prototypes.py, which
instead replicates WaffleCLIP's original multi-descriptor-averaging 'waffle' mode.

Example:
    real:    "Breast tumor epithelial tissue at 20x magnification, showing densely
              cellular malignant epithelial proliferation arranged in irregular
              glands, nests, cords, or solid sheets, with disrupted architecture,
              variable gland formation, and infiltrative growth within fibrous or
              desmoplastic stroma."
    waffled: "Breast tumor epithelial tissue at 20x magnification, showing woven
              gray solid rally arranged in coastal winds, hulls, mixes, or hazel
              drift, with corked column, distant beam runway, and involved supply
              within molten or countered result."

Input: a JSON file in the same nested format as camelyon_description.json:
    { "0": { "classnames": { "<ClassId>": ["<single long description>"], ... },
             "templates": ["CLASSNAME"] } }

Output: an .npy of shape [embedding_dim, num_classes] (one prompt per class, no
averaging -- matches how the real descriptions are used, templates=["CLASSNAME"]),
plus, optionally, a waffled JSON in the same schema as the input (--out_json) so it's
a drop-in swap for whatever consumes camelyon_description.json.

Usage:
    python generate_waffle_long_description.py --seed 0 \
        --real_descriptions_json camelyon_description.json \
        --out_path local_data/prompts/CONCH/CAMELYON16_waffle_description.npy \
        --out_json camelyon_description_waffle.json
"""

import os
import re
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer


DEFAULT_STRUCTURAL_WORDS = {
    "a", "an", "the", "at", "of", "in", "on", "or", "and", "with", "within", "without",
    "showing", "arranged", "composed", "embedded", "exhibits", "exhibiting",
    "relatively", "including", "such", "as", "by", "to", "from", "into",
    "tissue", "magnification", "x", "×",
}

TOKEN_RE = re.compile(r"^([^\w×]*)([\w×]+(?:['’-][\w×]+)*)([^\w×]*)$", re.UNICODE)


def split_token(tok):
    m = TOKEN_RE.match(tok)
    if m is None:
        return "", tok, ""
    return m.groups()


def is_protectable(core, structural_words):
    """Numbers, magnification tokens (20x, 20×), and structural/function words are
    kept verbatim rather than waffled."""
    if core.lower() in structural_words:
        return True
    stripped = core.lower().replace("x", "").replace("×", "")
    if stripped.isdigit():
        return True
    if not core.isalpha() and not any(c.isalpha() for c in core):
        return True
    return False


def pick_replacement(core, word_list, rng):
    target_len = len(core)
    candidates = [w for w in word_list if abs(len(w) - target_len) <= 2]
    if not candidates:
        candidates = word_list
    word = rng.choice(candidates)
    if core.isupper():
        word = word.upper()
    elif core[0].isupper():
        word = word[0].upper() + word[1:]
    return word


def waffle_description(text, base_classname, word_list, structural_words, rng):
    """Replaces content words in `text` with random words, protecting the leading
    base_classname phrase, structural/function words, numbers, and punctuation."""
    tokens = text.split(" ")

    # Figure out how many leading whitespace-split tokens the base_classname phrase
    # covers, so we can protect them regardless of the structural-word set.
    n_protected_leading = 0
    if base_classname and text.lower().startswith(base_classname.lower()):
        n_protected_leading = len(base_classname.split(" "))

    out_tokens = []
    for i, tok in enumerate(tokens):
        if tok == "":
            out_tokens.append(tok)
            continue
        prefix, core, suffix = split_token(tok)
        if i < n_protected_leading or core == "" or is_protectable(core, structural_words):
            out_tokens.append(tok)
            continue
        replacement = pick_replacement(core, word_list, rng)
        out_tokens.append(f"{prefix}{replacement}{suffix}")

    return " ".join(out_tokens)


@torch.no_grad()
def encode_single(model, tokenizer, text, device):
    token_ids = tokenize(tokenizer, [text]).to(device)
    embedding = model.encode_text(token_ids)
    embedding = F.normalize(embedding, dim=-1)
    return embedding.squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes_id_order", nargs="+", default=["Normal", "Tumor"])
    parser.add_argument("--base_classnames", nargs="+", default=None,
                         help="Leading classname phrase per class, protected from "
                              "waffling (e.g. 'Normal breast tissue' 'Breast tumor "
                              "epithelial tissue'). Defaults to the first few words of "
                              "each real description, split on the first comma.")
    parser.add_argument("--real_descriptions_json", default="local_data/prompts/templates/camelyon_description.json",
                         help="Same nested schema as camelyon_description.json: "
                              "{'0': {'classnames': {ClassId: [description]}, "
                              "'templates': ['CLASSNAME']}}.")
    parser.add_argument("--word_list_path", default="local_data/word_lists/broad_word_list.txt")
    parser.add_argument("--structural_words", nargs="*", default=[],
                         help="Extra words (beyond the built-in defaults) to protect "
                              "from waffling, e.g. domain terms you still want kept.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin")
    parser.add_argument("--out_path", default="local_data/prompts/CONCH/CAMELYON16_waffle_description.npy")
    parser.add_argument("--out_json", default=None,
                         help="Optional: also write the waffled descriptions as a JSON "
                              "in the same schema as --real_descriptions_json (a drop-in "
                              "swap for anything that consumes camelyon_description.json).")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(args.real_descriptions_json, "r") as f:
        real_json = json.load(f)
    top_key = list(real_json.keys())[0]
    classnames_block = real_json[top_key]["classnames"]
    templates = real_json[top_key]["templates"]
    assert templates == ["CLASSNAME"], (
        f"Expected templates == ['CLASSNAME'] (description used verbatim) in "
        f"{args.real_descriptions_json}, got {templates}"
    )

    real_descriptions = {}
    for cls_id in args.classes_id_order:
        assert cls_id in classnames_block, f"{cls_id} not found in {args.real_descriptions_json}"
        entries = classnames_block[cls_id]
        assert len(entries) == 1, (
            f"Expected exactly one long description for {cls_id}, found {len(entries)}"
        )
        real_descriptions[cls_id] = entries[0]

    if args.base_classnames is None:
        base_classnames = [real_descriptions[c].split(",")[0] for c in args.classes_id_order]
    else:
        assert len(args.base_classnames) == len(args.classes_id_order)
        base_classnames = args.base_classnames
    base_classname_by_class = dict(zip(args.classes_id_order, base_classnames))

    with open(args.word_list_path, "r") as f:
        word_list = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(word_list)} words from {args.word_list_path}")

    structural_words = set(DEFAULT_STRUCTURAL_WORDS) | {w.lower() for w in args.structural_words}

    waffled_by_class = {}
    for cls_id in args.classes_id_order:
        waffled_by_class[cls_id] = waffle_description(
            real_descriptions[cls_id], base_classname_by_class[cls_id],
            word_list, structural_words, rng,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.checkpoint_path)
    model = model.to(device).eval()
    tokenizer = get_tokenizer()

    zeroshot_weights = []
    for cls_id in args.classes_id_order:
        embedding = encode_single(model, tokenizer, waffled_by_class[cls_id], device)
        zeroshot_weights.append(embedding)
    prototypes = torch.stack(zeroshot_weights, dim=1).cpu().numpy().astype(np.float32)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, prototypes)
    print(f"Saved waffled-description prototypes {prototypes.shape} -> {args.out_path}\n")

    for cls_id in args.classes_id_order:
        print(f"--- {cls_id} ---")
        print(f"real:    {real_descriptions[cls_id]}")
        print(f"waffled: {waffled_by_class[cls_id]}\n")

    if args.out_json:
        out_obj = {
            "0": {
                "classnames": {cls_id: [waffled_by_class[cls_id]] for cls_id in args.classes_id_order},
                "templates": ["CLASSNAME"],
            }
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out_obj, f, indent=4)
        print(f"Wrote waffled description JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
