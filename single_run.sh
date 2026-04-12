#!/usr/bin/env bash
#SBATCH --job-name=QCV-final-run
#SBATCH --comment="run on test"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.seebode@campus.lmu.de
#SBATCH --partition=Krater
#SBATCH --nodes=1
#SBATCH -n 1

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

mkdir -p out/

source venv/bin/activate && echo "venv loaded"


SEED=$1

echo "Dispatching job"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 cdqbm_main.py \
  --path "out/" \
  --seed $SEED &


echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

mail -s "QCV-test-run-finished" m.seebode@campus.lmu.de
echo "Slurm execution done."
