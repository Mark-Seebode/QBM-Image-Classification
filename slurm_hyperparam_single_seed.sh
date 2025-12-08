#!/usr/bin/env bash
#SBATCH --job-name=CDQBM NEU-CLS
#SBATCH --comment="Running 1 seeds with hyperparameter optimization"
#SBATCH --partition=Krater
#SBATCH --nodes=1-1
#SBATCH -n 1

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

source venv/bin/activate && echo "venv loaded"

# Capture hyperparameters from command-line arguments
LEARNING_RATE=$1
BATCH_SIZE=$2
SAMPLE_COUNT=$3
SEED=$4
KERNEL_SIZE=$5
NUM_KERNELS=$6
SEQUENTIAL_LAYER_SIZES=$7
IS_RECURRENT_WEIGHTS=$8
RESTRICTED=${9}
ANNEALING_STEPS=${10}


# Dispatch each seed in parallel

echo "Dispatching job: seed $SEED with hyperparameters"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 cdqbm_main.py \
    --seed $SEED \
    --kernel_size $KERNEL_SIZE \
    --num_kernels $NUM_KERNELS \
    --sequential_layer_sizes $SEQUENTIAL_LAYER_SIZES \
    --is_recurrent_weights $IS_RECURRENT_WEIGHTS \
    --restricted $RESTRICTED \
    --annealing_steps $ANNEALING_STEPS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --sample_count $SAMPLE_COUNT \
    --data_set "NEU-CLS-64" \
    --load_path "out/" \
    --name "seed_$SEED"\
    --test_on_val "True" &

echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

echo "Slurm execution done."
