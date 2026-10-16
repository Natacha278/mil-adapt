#!/bin/bash
#SBATCH --account=rrg-josedolz
#SBATCH --job-name=text-topk-mil
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M             # all 399 bags of 512-d features are loaded, as in main.py
#SBATCH --time=0-01:00
#SBATCH --output=%x-%j.out

module load python/3.11 cuda openslide
module load gcc arrow/25.0.0
source ~/envs/mil-adapter/bin/activate

cd /project/rrg-josedolz/natgill/baselines/MIL-Adapter

python analyze_miladapter_attention_vs_witness_rate.py \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
        --xml_dir $SCRATCH/camelyon16/annotations \
        --text_prototypes local_data/prompts/CONCH/CAMELYON16.npy \
        --adapter TaskRes --k_shots 4 --seeds 0 1 2 3 4 5 6 7 8 9 \
        --out_prefix analysis/analysis_miladapter_attention/TaskRes_k4_class

