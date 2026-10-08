#!/bin/bash
#SBATCH --account=rrg-josedolz
#SBATCH --job-name=mil-adapter-camelyon16-lesion-size
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M             # higher than tumor_patch_acc.sh: keeps per-patch
                                 # occupancy + (with --probe) projected features in RAM
#SBATCH --time=0-00:30
#SBATCH --output=%x-%j.out


module load python/3.11 cuda openslide
module load gcc arrow/25.0.0
source ~/envs/mil-adapter/bin/activate

cd /project/rrg-josedolz/natgill/baselines/MIL-Adapter

# Sanity-check the geometry first -- needs no GPU, no data, no CONCH (~2 s).
python analyze_zeroshot_score_vs_lesion_size.py --self_test || exit 1

python analyze_zeroshot_score_vs_lesion_size.py \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16.npy \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --out_dir analysis/analysis_lesion_size_class \
    --save_patches

# Notes
# -----
# --probe adds the supervised patch-level linear probe as a CEILING reference. It uses
#   patch labels the MIL setting does not have, so it is a diagnostic, never a result:
#   if the zero-shot AUC falls in the small-lesion bins but the probe AUC does not, the
#   features encode small lesions and the TEXT PROMPT is what fails; if both fall, the
#   ENCODER is the limit and no pooling operator can fix it.
# --mpp_level0 defaults to 0.243 um/px (3DHistech). CAMELYON16 also contains Hamamatsu
#   slides at ~0.226. Diameters scale linearly with it, so pass the right value per
#   scanner if you need exact mm (the bin ORDERING is unaffected).
# Swap --text_prototypes to compare prompt banks (CAMELYON16.npy, *_class_aug.npy,
#   *_description_aug.npy, *_waffle*.npy) -- the per-lesion CSV is the natural place to
#   see which prompt bank holds up at small lesion size, which the pooled accuracy
#   number in analyze_zeroshot_patch_accuracy.py cannot show.
# Add --no_score for a login-node, geometry-only run (lesion sizes, patch counts,
#   occupancy, beta_hat; no CONCH, no GPU).
