#!/bin/bash
 
#SBATCH --nodes=2                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=36         # the number of tasks/processes per node
#SBATCH --cpus-per-task=1         # the number cpus per task
#SBATCH --partition=normal          # on which partition to submit the job
#SBATCH --time=01:00:00             # the max wallclock time (time limit your job will run)
#SBATCH --job-name=testParallelMATLAB    # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=elio.balestrieri@gmail.com # your mail address
#SBATCH --array=1-2
 
# LOAD MODULES HERE IF REQUIRED
module load matlab/R2022a

# DEFINE ARRAY OF INPUTS 
# these are parsed from the JobList.csv and submitted to parallel scripts
THISJOB=$(sed -n $SLURM_ARRAY_TASK_ID'p' JobList.csv | cut -d ',' -f2)

# START THE APPLICATION
matlab -nodisplay -nosplash -nodesktop -r "x='$THISJOB'; NodeLevelFunc(x); exit;"
