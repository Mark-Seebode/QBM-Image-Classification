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
CONV_LEARNING_RATE=$2
BATCH_SIZE=$3
SAMPLE_COUNT=$4
SEED=$5
KERNEL_SIZE=$6
NUM_KERNELS=$7
RESTRICTED=$8
#ANNEALING_STEPS=${8}
# Collect remaining args (0–3 values) as list
shift 8
SEQ_SIZES=("$@")    # now SEQ_SIZES is an array of the remaining args


# Dispatch each seed in parallel

echo "Dispatching job: seed $SEED with hyperparameters"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 cdqbm_main.py \
    --seed $SEED \
    --kernel_size $KERNEL_SIZE \
    --num_kernels $NUM_KERNELS \
    --learning_rate $LEARNING_RATE \
    --conv_learning_rate $CONV_LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --sample_count $SAMPLE_COUNT \
    --save "out/slurm/" \
    --name "seed_$SEED"\
    --test_on_val "True" \
    --restricted $RESTRICTED \
    --sequential_layer_sizes "${SEQ_SIZES[@]}" &

echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

echo "Slurm execution done."
