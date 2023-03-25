#!/bin/bash
 
#SBATCH --nodes=2                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node
#SBATCH --cpus-per-task=30         # the number cpus per task
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=24:00:00             # the max wallclock time (time limit your job will run)
#SBATCH --job-name=modelCompare    # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=elio.balestrieri@gmail.com # your mail address
#SBATCH --array=1-2
 
# LOAD MODULES HERE IF REQUIRED
module load palma/2021b
module load GCC/11.2.0
module load OpenMPI/4.1.1
module load scikit-learn/1.0.1
module load dask


# DEFINE ARRAY OF INPUTS 
EXPCONDS=$(sed -n $SLURM_ARRAY_TASK_ID'p' expconds_shlauncher.csv | cut -d ',' -f2)

# START THE APPLICATION
python HPC_betweensubjs_mdl_compare.py $EXPCONDS
