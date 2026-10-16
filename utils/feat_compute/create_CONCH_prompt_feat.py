import json
import numpy as np
import torch

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
from conch.downstream.zeroshot_path import zero_shot_classifier

# ---- config ----
checkpoint_path = "/project/rrg-josedolz/natgill/weights/conch_v1/pytorch_model.bin"  # adjust to wherever your TRIDENT setup downloaded it
template_json = "/project/rrg-josedolz/natgill/baselines/MIL-Adapter/local_data/prompts/templates/camelyon_description.json"
out_path = "local_data/prompts/CONCH/CAMELYON16_description.npy"

# must match classes_id order in utils/utils.py get_project_data() for CAMELYON16
classes_id_order = ["Normal", "Tumor"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- load model + tokenizer ----
model, _ = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=checkpoint_path)
model = model.to(device).eval()
tokenizer = get_tokenizer()

# ---- load prompt templates ----
with open(template_json) as f:
    prompt_data = json.load(f)

block = prompt_data["0"]  # single prompt set; adjust key if you add more later
templates = block["templates"]
classnames_by_class = block["classnames"]

missing = [c for c in classes_id_order if c not in classnames_by_class]
if missing:
    raise ValueError(f"classes_id {missing} not found in JSON 'classnames' keys: {list(classnames_by_class.keys())}")

# enforce class order explicitly, regardless of JSON dict order
classnames = [classnames_by_class[c] for c in classes_id_order]

print(f"Classes (in order): {classes_id_order}")
for c, names in zip(classes_id_order, classnames):
    print(f"  {c}: {len(names)} classname variants")
print(f"{len(templates)} templates")

# ---- generate zero-shot text prototypes ----
zeroshot_weights = zero_shot_classifier(model, classnames, templates, tokenizer=tokenizer, device=device)
# shape: [embedding_dim, num_classes]
print("Output shape:", tuple(zeroshot_weights.shape))

embeddings = zeroshot_weights.cpu().numpy().astype(np.float32)
np.save(out_path, embeddings)
print(f"Saved to {out_path}")