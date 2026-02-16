#!/usr/bin/env bash
#SBATCH --job-name=CDQBM-QA
#SBATCH --comment="Neu-CLS-64"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.seebode@campus.lmu.de
#SBATCH --partition=Krater
#SBATCH --nodes=1
#SBATCH -n 1

export TMPDIR=$HOME/tmp
mkdir -p $TMPDIR

mkdir -p out/slurm/

source venv/bin/activate && echo "venv loaded"




echo "Dispatching job"
srun --exclusive --ntasks=1 --nodes=1 -c 1 python3 qa_hyperparam.py --path "out/"

echo -e "\tWaiting for Job completion."
wait

echo -e "\nAll jobs finished"

mail -s "QCV-test-run-finished" m.seebode@campus.lmu.de
echo "Slurm execution done."