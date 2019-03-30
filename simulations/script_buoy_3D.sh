#!/bin/bash
# Job name
#PBS -N buoy_3D_test
# Time required in hh:mm:ss
#PBS -l walltime=00:10:00
# Resource requirements
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=30999Mb

module load anaconda3/personal
module load gcc
module load mpi
export I_MPI_CC=gcc
export I_MPI_CXX=g++

echo Working Directory is "$(pwd)"

mkdir -p data

source activate parcelsvenv

python $HOME/packages/turbulence-patchiness-sims/simulations/script_3D_buoyancy_offline_HPC.py

mv data/* $EPHEMERAL/results/

source deactivate
