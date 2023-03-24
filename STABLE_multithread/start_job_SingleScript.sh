#!/bin/bash
 
#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node
#SBATCH --cpus-per-task=30          # the number cpus per task
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=24:00:00             # the max wallclock time (time limit your job will run)
 
 
#SBATCH --job-name=decodeFreqBands  # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=elio.balestrieri@gmail.com # your mail address
 
# LOAD MODULES HERE IF REQUIRED
module load palma/2021b
module load GCC/11.2.0
module load OpenMPI/4.1.1
module load scikit-learn/1.0.1
module load dask

# START THE APPLICATION
python HPC_withinsubjs_mdl_compare.py 
