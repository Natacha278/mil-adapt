#!/bin/bash
#SBATCH --account=rrg-josedolz
#SBATCH --job-name=mil-adapter-camelyon16-conch
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000M              # adjust based on the `du -sh` output above
#SBATCH --time=0-00:10
#SBATCH --output=%x-%j.out


module load python/3.11 cuda openslide
module load gcc arrow/25.0.0
source ~/envs/mil-adapter/bin/activate

cd /project/rrg-josedolz/natgill/baselines/MIL-Adapter

# python analyze_tumor_pct_accuracy.py \
#     --adapter TaskRes --init ZS --aggregator ABMIL \
#     --k_shots 8 --seed 0 \
#     --xml_dir $SCRATCH/camelyon16/annotations \
#     --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
#     --text_prototypes local_data/prompts/CONCH/CAMELYON16.npy \
#     --out_dir ./analysis

python analyze_tumor_pct_accuracy.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --adapter TaskRes --init ZS --aggregator ABMIL \
    --k_shots 8 --seed 1 --n_bins 8 \
    --out_dir ./analysis_tum_ratio \
    --out_prefix k8_s1_TaskRes

