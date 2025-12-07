#!/usr/bin/env bash
#SBATCH --job-name=QCV-hyper-run
#SBATCH --comment="breast mnist hyperparameter search"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.seebode@campus.lmu.de
#SBATCH --partition=All
#SBATCH --nodes=1
#SBATCH -n 1

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

source venv/bin/activate && echo "venv loaded"




echo "Dispatching job"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 slurm_hyperparam.py --path "QuCUN_2hnodes/"

echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

mail -s "QCV-test-run-finished" m.seebode@campus.lmu.de
echo "Slurm execution done."
