#!/usr/bin/env bash
#SBATCH --job-name=CDQBM-NEU-CLS
#SBATCH --comment="Running 1 seeds with hyperparameter optimization"
#SBATCH --partition=Krater
#SBATCH --nodes=1-1
#SBATCH -n 1

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

mkdir -p out/slurm

source venv/bin/activate && echo "venv loaded"

# Capture hyperparameters from command-line arguments
LEARNING_RATE=$1
N_HIDDEN_NODES=$2
BATCH_SIZE=$3
SAMPLE_COUNT=$4
SEED=$5

# Dispatch each seed in parallel

echo "Dispatching job: seed $SEED with hyperparameters"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 qbm_main.py \
    --seed $SEED \
    --hnodes $N_HIDDEN_NODES \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --sample_count $SAMPLE_COUNT \
    --load_path "out/slurm/" \
    --name "seed_$SEED"&

echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

echo "Slurm execution done."
