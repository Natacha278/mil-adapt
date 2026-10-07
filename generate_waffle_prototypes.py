"""
Generates CONCH text prototypes for CAMELYON16 using WaffleCLIP's "waffle" random-word
method (Roth et al., "Waffling around for Performance: Visual Classification with Random
Words and Broad Concepts", ICCV 2023 — github.com/ExplainableML/WaffleCLIP), instead of
the meaningful LLM-written classnames/descriptions in camelyon_prompt_templates.json.

Purpose: an ablation to test whether the descriptive content of the CAMELYON16 prompts
matters at all, or whether CONCH's zero-shot/adapter performance is mostly driven by the
extra degrees of freedom that come from averaging several distinct text embeddings per
class ("broad concepts help regardless of relevance" — the paper's core finding on
natural-image benchmarks).

Faithfully replicates WaffleCLIP's 'waffle' mode from waffle_tools.py:
  - For each class, draws `waffle_count` random REAL-WORD descriptors from a broad,
    domain-agnostic word list (not pathology-specific), plus `waffle_count` random
    CHARACTER-NOISE descriptors matched in length/spacing to the classnames.
  - All classes share the exact same randomly-sampled descriptor set (only the classname
    token differs) — this is WaffleCLIP's "match_key" behaviour: the randomness is shared
    across classes, not independently drawn per class.
  - Descriptors are wrapped with structured_descriptor_builder:
        "{pre}{label_before}{classname}{separator}{which has|is <descriptor>}{label_after}"
    which for the pathology setting defaults to something like:
        "a photomicrograph of a tumor tissue, which has zebra."

Text embeddings are produced with CONCH's own text encoder (not CLIP's), following the
same averaging convention as CONCH's zero_shot_classifier / the existing
camelyon_prompt_templates.json prototypes: embed each descriptor sentence independently,
L2-normalize, mean over all descriptors for a class, then renormalize. This makes the
output directly comparable (same shape, same downstream usage via --text_prototypes) to
local_data/prompts/CONCH/CAMELYON16.npy.

Usage:
    python generate_waffle_prototypes.py --seed 0 --waffle_count 15 \
        --out_path local_data/prompts/CONCH/CAMELYON16_waffle.npy

The output can be A/B compared against the real-description prototypes with
visu_ABMIL.py / analyze_tumor_pct_accuracy.py by pointing --text_prototypes at it.
"""

import os
import json
import random
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer


# ----------------------------- WaffleCLIP mechanics -----------------------------

def wordify(string):
    return string.replace('_', ' ')


def make_descriptor_sentence(descriptor):
    # matches waffle_tools.make_descriptor_sentence exactly
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or \
            descriptor.startswith('typically') or descriptor.startswith('may') or \
            descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"


def modify_descriptor(descriptor, apply_changes):
    if apply_changes:
        return make_descriptor_sentence(descriptor)
    return descriptor


def build_structured_descriptor(item, cls, pre_descriptor_text, label_before_text,
                                 descriptor_separator, label_after_text,
                                 apply_descriptor_modification):
    return (f"{pre_descriptor_text}{label_before_text}{wordify(cls)}"
            f"{descriptor_separator}{modify_descriptor(item, apply_descriptor_modification)}"
            f"{label_after_text}")


def generate_waffle_descriptors(class_names, word_list, waffle_count, rng):
    """
    Replicates waffle_tools.py's mode == 'waffle' block:
      - avg_num_words / avg_word_length / num_spaces / num_chars are derived from the
        classnames themselves (so the noise strings look roughly like the classnames).
      - character_list (for the noise descriptors) is drawn from the letters that appear
        in the classnames, mirroring their "(Lazy solution)" use of the description list.
      - the SAME randomly-sampled base_word/noise_word pairs are shared across all classes
        (WaffleCLIP's match_key behaviour: one class's random draw is copied to the rest,
        with only the classname substring swapped).

    Returns: dict {class_name: [descriptor_str, ...]} — waffle_count*2 raw descriptor
    strings (not yet wrapped in the full template) per class, IDENTICAL across classes.
    """
    wordified = [wordify(c) for c in class_names]

    avg_num_words = int(max(round(np.mean([len(w.split(' ')) for w in wordified])), 1))
    avg_word_length = int(round(np.mean([np.mean([len(y) for y in w.split(' ')]) for w in wordified])))
    truncated_word_list = [w[:avg_word_length] for w in word_list if len(w) > 0]

    character_list = sorted(set(''.join(wordified).replace(' ', '')))
    if len(character_list) == 0:
        character_list = list('abcdefghijklmnopqrstuvwxyz')

    num_spaces = int(round(np.mean([w.count(' ') for w in wordified]))) + 1
    num_chars = int(np.ceil(np.mean([max(len(y) for y in w.split(' ')) for w in wordified])))
    num_chars += num_spaces - (num_chars % num_spaces if num_spaces else 0)

    sample_key = ''
    for s in range(num_spaces):
        for _ in range(num_chars // num_spaces):
            sample_key += 'a'
        if s < num_spaces - 1:
            sample_key += ' '

    # Draw ONE shared set of descriptors (this directly produces WaffleCLIP's
    # match_key outcome without the wasteful per-class-then-overwrite step).
    shared_descriptors = []
    for _ in range(waffle_count):
        base_word = ''
        for a in range(avg_num_words):
            base_word += rng.choice(truncated_word_list)
            if a < avg_num_words - 1:
                base_word += ' '
        shared_descriptors.append(base_word)

        noise_word = ''
        for c in sample_key:
            if c != ' ':
                noise_word += rng.choice(character_list)
            else:
                noise_word += ', '
        shared_descriptors.append(noise_word)

    return {c: list(shared_descriptors) for c in class_names}


# ----------------------------- CONCH text encoding -----------------------------

@torch.no_grad()
def encode_and_average(model, tokenizer, texts, device):
    """Embeds each text independently, L2-normalizes, mean-pools, renormalizes.
    Mirrors CONCH's zero_shot_classifier averaging convention."""
    token_ids = tokenize(tokenizer, texts).to(device)
    embeddings = model.encode_text(token_ids)
    embeddings = F.normalize(embeddings, dim=-1)
    class_embedding = embeddings.mean(dim=0)
    class_embedding = class_embedding / class_embedding.norm()
    return class_embedding


def build_waffle_prototypes(model, tokenizer, device, classes_id_order, descriptors_by_class,
                             display_name_by_class, pre_descriptor_text, label_before_text,
                             descriptor_separator, label_after_text, apply_descriptor_modification):
    zeroshot_weights = []
    all_texts_per_class = {}
    for cls in classes_id_order:
        texts = [
            build_structured_descriptor(
                item, display_name_by_class[cls], pre_descriptor_text, label_before_text,
                descriptor_separator, label_after_text, apply_descriptor_modification,
            )
            for item in descriptors_by_class[cls]
        ]
        all_texts_per_class[cls] = texts
        class_embedding = encode_and_average(model, tokenizer, texts, device)
        zeroshot_weights.append(class_embedding)

    # [embedding_dim, num_classes] — matches CONCH zero_shot_classifier's output layout
    # and local_data/prompts/CONCH/CAMELYON16.npy's existing shape.
    prototypes = torch.stack(zeroshot_weights, dim=1).cpu().numpy().astype(np.float32)
    return prototypes, all_texts_per_class


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes_id_order", nargs="+", default=["Normal", "Tumor"])
    parser.add_argument("--base_classnames", nargs="+", default=None,
                         help="One base classname per entry in --classes_id_order, e.g. "
                              "'normal tissue' 'tumor tissue'. Defaults to "
                              "'<class> tissue' (lowercased) for each class.")
    parser.add_argument("--word_list_path", default="local_data/word_lists/broad_word_list.txt",
                         help="Plain text file, one word per line. Domain-agnostic broad "
                              "vocabulary (WaffleCLIP's own word_list.pkl, not pathology-"
                              "specific) — a 573-word subsample of the paper's original "
                              "4010-word list is shipped by default; swap in the full list "
                              "here if you have it.")
    parser.add_argument("--waffle_count", type=int, default=15,
                         help="Descriptors per class (WaffleCLIP default M=15). Each count "
                              "contributes one random-word + one random-character descriptor, "
                              "so the class prototype averages 2*waffle_count embeddings.")
    parser.add_argument("--pre_descriptor_text", default="")
    parser.add_argument("--label_before_text", default="a photomicrograph of a ",
                         help="WaffleCLIP's ImageNet default is 'A photo of a '; adapted "
                              "here to match the histopathology phrasing used in "
                              "camelyon_prompt_templates.json.")
    parser.add_argument("--label_after_text", default=".")
    parser.add_argument("--descriptor_separator", default=", ")
    parser.add_argument("--no_apply_descriptor_modification", action="store_true",
                         help="If set, use the raw descriptor text verbatim instead of "
                              "wrapping it as 'which has <descriptor>' / 'which is <descriptor>'.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_path", default="/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin" )
    parser.add_argument("--out_path", default="local_data/prompts/CONCH/CAMELYON16_waffle.npy")
    parser.add_argument("--dump_texts_path", default=None,
                         help="Optional: also dump the generated (classname, prompt list) "
                              "texts as JSON, for a sanity check of what was actually encoded.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    if args.base_classnames is None:
        base_classnames = [f"{c.lower()} tissue" for c in args.classes_id_order]
    else:
        assert len(args.base_classnames) == len(args.classes_id_order), \
            "--base_classnames must have one entry per --classes_id_order"
        base_classnames = args.base_classnames

    with open(args.word_list_path, "r") as f:
        word_list = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(word_list)} words from {args.word_list_path}")

    descriptors_by_class = generate_waffle_descriptors(
        base_classnames, word_list, args.waffle_count, rng
    )
    # map back from base_classname -> class id (Normal/Tumor)
    descriptors_by_class = {
        cls_id: descriptors_by_class[base] for cls_id, base in zip(args.classes_id_order, base_classnames)
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=args.checkpoint_path)
    model = model.to(device).eval()
    tokenizer = get_tokenizer()

    display_name_by_class = dict(zip(args.classes_id_order, base_classnames))

    prototypes, all_texts_per_class = build_waffle_prototypes(
        model, tokenizer, device,
        classes_id_order=args.classes_id_order,
        descriptors_by_class={cls_id: descriptors_by_class[cls_id] for cls_id in args.classes_id_order},
        display_name_by_class=display_name_by_class,
        pre_descriptor_text=args.pre_descriptor_text,
        label_before_text=args.label_before_text,
        descriptor_separator=args.descriptor_separator,
        label_after_text=args.label_after_text,
        apply_descriptor_modification=not args.no_apply_descriptor_modification,
    )

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    np.save(args.out_path, prototypes)
    print(f"Saved waffle prototypes {prototypes.shape} -> {args.out_path}")

    for cls_id in args.classes_id_order:
        print(f"\n--- {cls_id} ({len(all_texts_per_class[cls_id])} descriptors) ---")
        for t in all_texts_per_class[cls_id][:6]:
            print(f"  {t}")
        if len(all_texts_per_class[cls_id]) > 6:
            print(f"  ... ({len(all_texts_per_class[cls_id]) - 6} more)")

    if args.dump_texts_path:
        os.makedirs(os.path.dirname(args.dump_texts_path) or ".", exist_ok=True)
        with open(args.dump_texts_path, "w") as f:
            json.dump(all_texts_per_class, f, indent=2)
        print(f"\nDumped full prompt texts -> {args.dump_texts_path}")


if __name__ == "__main__":
    main()
