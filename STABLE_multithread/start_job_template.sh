#!/bin/bash
 
#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node
#SBATCH --cpus-per-task=22          # the number cpus per task
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=10:00:00             # the max wallclock time (time limit your job will run)
#SBATCH --output=../computed_features/freqBands.dat
 
 
#SBATCH --job-name=decodeFreqBands  # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=ebalestr@uni-muenster.de # your mail address
 
# LOAD MODULES HERE IF REQUIRED
module load palma/2021b
module load GCC/11.2.0
module load OpenMPI/4.1.1
module load scikit-learn/1.0.1
module load dask

# START THE APPLICATION
python HPC_freqbands_classifier.py 
