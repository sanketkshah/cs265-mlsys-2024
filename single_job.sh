#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --gres gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem 64G               # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-00:30:00             # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p gpu_requeue,seas_gpu,gpu        # Partition to submit to
#SBATCH -o myoutput_%A_%a.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%A_%a.err  # File to which STDERR will be written, %j inserts jobid

cd /n/netscratch/tambe_lab/Lab/sshah/cs265-mlsys-2024/
source .venv/bin/activate

tail -n+$SLURM_ARRAY_TASK_ID experiments.txt | head -1 | bash