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

# normal_010 
# Micro: tumor_001 tumor_002
# Macro: tumor_004 tumor_009
# ITC: tumor_008 tumor_017

python visu_ABMIL.py \
    --slide_id tumor_001 \
    --adapter TaskRes --init ZS \
    --k_shots 16 --seed 0 \
    --wsi_dir $SCRATCH/camelyon16/images \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat 


python visu_zs_classes.py \
        --slide_ids normal_081 normal_024 tumor_001 tumor_004 tumor_009 tumor_008 tumor_016 tumor_050 tumor_026\
        --class_json local_data/prompts/templates_attribs/all_attribs_v3.json \
        --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
        --wsi_dir $SCRATCH/camelyon16/images --xml_dir $SCRATCH/camelyon16/annotations \
        --exp_name "" \
        --out_dir ./visualizations/zs_classes_all_attribs_v3

    
    