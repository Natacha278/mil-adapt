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

python -m utils.text_topk_mil || exit 1        # gradient check + synthetic benchmark (~2 s)

python run_text_topk_mil.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --topk 1 3 --lam 0.01 0.1 1 --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --lesion_csv analysis/analysis_lesion_size_descr/CAMELYON16_CONCH_zs_per_lesion.csv \
    --calibrate_zero_shot \
    --out_dir analysis/analysis_text_topk

# Six configurations (K x lambda) make the comparison figures busy; once the best one is
# known, rerun with only it, e.g. --topk 1 --lam 0.1, for the presentation figures.

    # --baseline "MIL-Adapter (TaskRes+ABMIL)=analysis/analysis_auc/CAMELYON16_CONCH_TaskRes_ABMIL_k16_per_slide.csv" \
    #            "MI-Zero top-1=analysis/analysis_mizero/CAMELYON16_MIZERO_logit_k1_per_slide.csv" \

python run_text_topk_mil.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --topk 3 --lam 0.1 --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --backend torch \
    --threshold zero conformal --alpha 0.1 0.12 0.18 \
    --calibrate_zero_shot --out_dir analysis/analysis_text_topk_conformal_k3


python run_text_topk_mil.py --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --topk 3 --lam 0.1 --k_shots 4 --seeds 0 1 2 3 4 5 6 7 8 9 --backend torch \
    --neg_weight 0 0.3 1 --neg_m 32 --neg_margin 10 --threshold zero conformal --alpha 0.12 \
    --out_dir analysis/analysis_text_topk4_rank

python run_text_topk_mil.py --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --topk 3 --lam 0.1 --k_shots 4 --seeds 0 1 2 3 4 5 6 7 8 9 --backend torch \
    --neg_weight 0 0.3 1 --neg_m 32 --neg_margin 10 --threshold zero\
    --out_dir analysis/analysis_text_topk4_rank



python analyze_sparse_detection_vs_witness_rate.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --k_shots 4 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out_prefix analysis/analysis_sparse/sparse_k4
    # --alpha0 0.01


python analyze_linear_probe_vs_witness_rate.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --k_shots 4 8 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --topk 3 --lam 0.1 --neg_weight 1 \
    --out_prefix analysis/analysis_ceiling/ceiling

python run_text_topk_mil.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --concepts local_data/prompts/CONCH/CAMELYON16_concepts_v2.npz --tau 1 \
    --topk 3 --lam 0.1 --k_shots 4 --seeds 0 1 2 3 4 5 6 7 8 9 --backend torch \
    --neg_weight 0 1 --neg_m 32 --neg_margin 10 --threshold zero \
    --out_dir analysis/analysis_text_topk4_mc2

python analyze_text_topk_vs_witness_rate.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --concepts local_data/prompts/CONCH/CAMELYON16_concepts.npz --tau 1 \
    --topk 3 --lam 0.1 --neg_weight 1 --k_shots 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --out_prefix analysis/analysis_text_topk/mechanism_mc_K3_lam0.1_neg1



python -m utils.witness_pseudolabel   # self-test
python run_witness_pseudolabel.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --topk 3 --lam 0.1 --neg_weight 1 --neg_m 32 --neg_margin 10 \
    --rounds 1 3 --patch_weight 0.3 1 --k_shots 4 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --out_dir analysis/analysis_pseudolabel_k4

python analyze_linear_probe_vs_witness_rate.py \
    --folder /project/rrg-josedolz/natgill/data/camelyon16_milformat \
    --xml_dir $SCRATCH/camelyon16/annotations \
    --text_prototypes local_data/prompts/CONCH/CAMELYON16_description.npy \
    --k_shots 4 8 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
    --topk 3 --lam 0.1 --neg_weight 1 --pl_rounds 3 --pl_patch_weight 1 \
    --out_prefix analysis/analysis_ceiling/ceiling_pl
