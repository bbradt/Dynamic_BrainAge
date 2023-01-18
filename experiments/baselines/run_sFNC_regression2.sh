#!/bin/bash
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -p qTRDHM
#SBATCH --mem=20gb
#SBATCH -e /data/users2/mduda/scripts/brainAge/LSTM_BrainAge/slurm/logs/sFNC_regression2.err
#SBATCH -o /data/users2/mduda/scripts/brainAge/LSTM_BrainAge/slurm/logs/sFNC_regression2.out
#SBATCH -t 7200
#SBATCH -J sFNC_brainage
#SBATCH -A trends53c17
#SBATCH --oversubscribe
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mduda@gsu.edu

sleep 7s
export OMP_NUM_THREADS=1
##export MODULEPATH=/apps/Compilers/modules-3.2.10/Debug-Build/Modules/3.2.10/modulefiles/
#NODE=$(hostname)

echo $HOSTNAME >&2
module load python
source /userapp/virtualenv/mduda/venv/bin/activate

python sFNC_regression2.py


sleep 7s
