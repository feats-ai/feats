#!/bin/bash

#obligatory parameters
#SBATCH -n 1
#SBATCH --mem-per-cpu=3000
#SBATCH -t 00:30:00

#recommended
#SBATCH -A projectXXXX
#SBATCH -J CCXJobArrayFEATS
#SBATCH -a 1-3
#SBATCH -o /ccx/simulations_sphere_15/%a/%x.out.%A_%a
#SBATCH -e /ccx/simulations_sphere_15/%a/%x.err.%A_%a

cd /work/projects/project02298/ccx/simulations_sphere_15/$SLURM_ARRAY_TASK_ID

LD_LIBRARY_PATH=/work/projects/projectXXXX/.conda/envs/ccx/lib
export OMP_NUM_THREADS=1
/work/projects/projectXXXX/bin/ccx-legacy sim
