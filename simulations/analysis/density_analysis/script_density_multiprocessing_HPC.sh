#!/bin/bash
# Job name
#PBS -N buoy_3D_test
# Time required in hh:mm:ss
#PBS -l walltime=24:00:00
# Resource requirements
#PBS -l select=1:ncpus=32:mem=20999:mpiprocs=1:ompthreads=1


module load anaconda3/personal
module load gcc
module load mpi
export I_MPI_CC=gcc
export I_MPI_CXX=g++

echo Working Directory is "$(pwd)"

SAVEDIR=$(pwd)
SAVEDIR=${SAVEDIR:(-9)}
SAVEDIR=${SAVEDIR:0:5}
SAVEDIR="results$SAVEDIR"
echo $SAVEDIR

mkdir -p $EPHEMERAL/$SAVEDIR

cd $EPHEMERAL/$SAVEDIR

echo New directory is "$(pwd)"

source activate parcelsvenv

python $HOME/packages/turbulence-patchiness-sims/simulations/analysis/analysis_tools/script_density_multiprocessing_HPC.py

source deactivate
