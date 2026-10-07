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

python visu_ABMIL_zs.py\
    --slide_id normal_010 \
    --adapter TaskRes --init ZS \
    --k_shots 16 --seed 0 \
    --wsi_dir $SCRATCH/camelyon16/images \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat 
    