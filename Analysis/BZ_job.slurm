#!/bin/bash
#SBATCH --job-name=fomula_search   # create a short name for your job
#SBATCH --nodes=20                # node count
#SBATCH --ntasks=4               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --array=0-14             # job array with index values 0 to 14 (length 15)
#SBATCH --time=24:00:00           # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=bz0014@princeton.edu

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_TASK_ID

module purge

Rscript ParallelJobTemplate.R $SLURM_ARRAY_TASK_ID
