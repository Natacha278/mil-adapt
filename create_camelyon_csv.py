import os
import re
import pandas as pd
from glob import glob

# ---- paths: adjust these two to your actual layout ----
features_dir = "/project/rrg-josedolz/natgill/data/camelyon16/trident/20x_512px_0px_overlap/features_conch_v1"
reference_csv = "/scratch/natgill/camelyon16/evaluation/reference.csv"  # confirm exact filename with `ls $SCRATCH/camelyon16/evaluation/`
out_csv = "local_data/csv/CAMELYON16.csv"  # run this from the MIL-Adapter repo root, or adjust path

# ---- 1. collect slide IDs from the feature files ----
slide_ids = sorted(
    os.path.splitext(os.path.basename(f))[0]
    for f in glob(os.path.join(features_dir, "*.h5"))
)
print(f"Found {len(slide_ids)} feature files")

# ---- 2. label train/normal and train/tumor slides directly from their filename prefix ----
records = {}
for sid in slide_ids:
    if sid.startswith("normal_"):
        records[sid] = "Normal"
    elif sid.startswith("tumor_"):
        records[sid] = "Tumor"
    elif sid.startswith("test_"):
        pass  # resolved from reference.csv below
    else:
        print(f"WARNING: unrecognized slide id prefix, skipping: {sid}")

# ---- 3. load reference.csv for test_* labels, auto-detecting header vs. no header ----
raw = pd.read_csv(reference_csv, header=None)
# if the first row's second column isn't a recognizable label, assume it's a header and reload
first_row_label = str(raw.iloc[0, 1]).strip().lower()
has_header = first_row_label not in ("tumor", "normal")
ref = pd.read_csv(reference_csv, header=0 if has_header else None)
ref.columns = [f"col{i}" for i in range(ref.shape[1])]  # normalize column names regardless of header text
print(f"reference.csv loaded with header={has_header}, {len(ref)} rows, columns={list(ref.columns)}")

label_map = {"tumor": "Tumor", "normal": "Normal"}
test_labels = {}
for _, row in ref.iterrows():
    name = str(row["col0"]).strip()
    name = re.sub(r"\.tif$", "", name, flags=re.IGNORECASE)  # strip extension if present
    label_raw = str(row["col1"]).strip().lower()
    if label_raw not in label_map:
        print(f"WARNING: unrecognized label '{row['col1']}' for {name}, skipping")
        continue
    test_labels[name] = label_map[label_raw]

for sid in slide_ids:
    if sid.startswith("test_"):
        if sid in test_labels:
            records[sid] = test_labels[sid]
        else:
            print(f"WARNING: no reference.csv entry found for {sid}")

# ---- 4. sanity checks before writing ----
missing = [sid for sid in slide_ids if sid not in records]
if missing:
    print(f"WARNING: {len(missing)} slide(s) have no label and will be EXCLUDED from the CSV: {missing[:10]}{'...' if len(missing) > 10 else ''}")

counts = pd.Series(records.values()).value_counts()
print("Label counts:\n", counts)

# ---- 5. write CSV in the format load_data() expects: case_id,WSI,GT ----
df = pd.DataFrame({
    "case_id": list(records.keys()),
    "WSI": list(records.keys()),
    "GT": list(records.values()),
})
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"Wrote {len(df)} rows to {out_csv}")