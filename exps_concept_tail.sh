module load python/3.11 cuda openslide
module load gcc arrow/25.0.0
source ~/envs/mil-adapter/bin/activate

cd /project/rrg-josedolz/natgill/baselines/MIL-Adapter

python encode_concepts.py --class_json local_data/prompts/templates/camelyon_concepttail_concepts.json \
    --out local_data/prompts/CONCH/CAMELYON16_concepttail.npz
python encode_text_bank.py --bank_json local_data/prompts/templates/pathology_text_bank.json \
    --out local_data/prompts/CONCH/pathology_text_bank.npz

python run_concept_tail_mil.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --concepts local_data/prompts/CONCH/CAMELYON16_concepttail.npz \
    --bank local_data/prompts/CONCH/pathology_text_bank.npz --rank 16 \
    --pool lse top1 top3 --lam_tail 0 1 --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --reference_text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --out_dir analysis/analysis_concept_tail