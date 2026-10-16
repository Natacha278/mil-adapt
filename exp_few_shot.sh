#!/bin/bash
#SBATCH --account=rrg-josedolz
#SBATCH --job-name=mil-adapter-nsclc-conch
#SBATCH --gpus-per-node=1          # mandatory — utils/trainer.py hardcodes .cuda(), CPU-only will crash
#SBATCH --cpus-per-task=4          # plenty for the sequential np.load() loop in load_data()
#SBATCH --mem=16000M              # load_data() loads EVERY NSCLC WSI's embedding into RAM up front, not just the few-shot subset — verify against your actual data size, see note below
#SBATCH --time=0-00:10             # generous buffer: small networks, batch_size=1, but --n_seeds defaults to 10 full train+validate cycles
#SBATCH --output=%x-%j.out

module load python/3.11 cuda
module load gcc arrow/25.0.0
source ~/envs/mil-adapter/bin/activate

cd /project/rrg-josedolz/natgill/baselines/MIL-Adapter

python main.py \
  --folder /project/rrg-josedolz/natgill/data \
  --project NSCLC \
  --encoder CONCH \
  --aggregator ABMIL \
  --adapter ZSMIL \
  --init random \
  --k_shots 8 \
  --epochs 20 \
  --lr 1e-3 \
  --n_seeds 10 \
  --output_results experiments_nsclc_conch.xlsx