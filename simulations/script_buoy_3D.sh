#!/bin/bash
# Job name
#PBS -N buoy_3D_test
# Time required in hh:mm:ss
#PBS -l walltime=00:10:00
# Resource requirements
#PBS -l select=1:ncpus=1:mpiprocs=1:ompthreads=1:mem=15999Mb
# Files to contain standard error and standard output
#PBS -o stdout
#PBS -e stderr
# Mail notification
#PBS -m ae
#PBS -M akc17@imperial.ac.uk

rm -f stdout* stderr*

module load gcc
module load mpi
export I_MPI_CC=gcc
export I_MPI_CXX=g++

source $HOME/pythondrake/bin/activate
mkdir -p data

# Start time.
echo Start time is `date` > data/time

python simulations/script_3D_buoyancy_offline.py

# End time.
echo end time is `date` >> data/time

# Copy results to ephemeral
cp -r data $EPHEMERAL