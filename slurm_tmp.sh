#!/bin/bash -l

#SBATCH -p course
#SBATCH -D ./
#SBATCH -t 3
#SBATCH --export=ALL

cores=${SLURM_CPUS_PER_TASK:-1} #Gets the requested number of cores when sbatch -c is used

export OMP_NUM_THREADS=$cores   #Exports the OMP_NUM_THREADS variable

# 1. Add code here to compile a certain program

# 2. Add code here to run the program

# 3. Then submit this script to the compute nodes. 
#    Instructions for how to do this are in the lab