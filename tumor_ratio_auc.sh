#!/bin/bash
#SBATCH --account=rrg-josedolz
#SBATCH --job-name=mil-adapter-tumor-ratio-auc
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=0-02:00
#SBATCH --output=%x-%j.out


module load python/3.11 cuda openslide
module load gcc arrow/25.0.0
source ~/envs/mil-adapter/bin/activate


cd /project/rrg-josedolz/natgill/baselines/MIL-Adapter

# Estimator checks first -- no GPU, no data, no CONCH, ~10 s each.
python analyze_tumor_ratio_miladapter.py --self_test || exit 1
python analyze_tumor_ratio_mizero.py     --self_test || exit 1

SEEDS="0 1 2 3 4 5 6 7 8 9"
PROMPTS=local_data/prompts/CONCH/CAMELYON16_description.npy

# ---- Arm 1: MIL-Adapter (mean-like aggregator) ------------------------------------
# Trains and scores internally over every seed. No change to main.py or trainer.py.
python analyze_tumor_ratio_miladapter.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --adapter TaskRes --aggregator ABMIL \
    --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out_dir analysis/analysis_auc

# ---- Arm 2: MI-Zero (order statistic) ----------------------------------------------
# Training-free. --k_shots/--seeds only exclude each seed's support slides, so the two
# arms are evaluated on exactly the same slides.
python analyze_tumor_ratio_mizero.py \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy\
    --topk 1 5 20 50 100 0.01 \
    --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out_dir analysis_mizero

# Optional: the same comparison with a second aggregator, to show the drop is a
# property of the pooling CLASS and not of ABMIL specifically.
# python analyze_tumor_ratio_miladapter.py \
#     --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
#     --text_prototypes $PROMPTS --adapter TaskRes --aggregator TransMIL \
#     --k_shots 16 --seeds $SEEDS --out_dir analysis_auc

# Reading the result
# ------------------
# Compare auc_gap (AUC_high - AUC_low across the beta=1/2 wall) and slope_u_per_decade
#   between the two <prefix>_summary.json files. Both arms use utils/tumor_ratio_metrics.py
#   verbatim, so the numbers are directly comparable.
# MI-Zero's auc_gap smaller at small K while its auc_high is lower -> order-statistic
#   signature, pooling regime survives as the explanation.
# auc_gap flat across K and matching MIL-Adapter -> pooling is NOT the mechanism; the
#   encoder is the suspect and analyze_zeroshot_score_vs_lesion_size.py decides it.
# spearman_score_vs_npatches_normal in the MI-Zero sweep: clearly positive at small K
#   means the statistic is partly reading slide size rather than tumor.
# The CI is slide-sampling only and the seed SD is the training component; they do not
#   combine, because every seed is evaluated on the same slides.
