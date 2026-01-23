#!/usr/bin/env bash
#SBATCH --job-name=CDQBM-NEU-CLS
#SBATCH --comment="Running 1 seeds with hyperparameter optimization"
#SBATCH --partition=Krater
#SBATCH --nodes=1-1
#SBATCH -n 1

mkdir -p out/slurm

source venv/bin/activate && echo "venv loaded"



# Dispatch each seed in parallel

echo "Dispatching job: seed $SEED with hyperparameters"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 qbm_main.py

echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

echo "Slurm execution done."
