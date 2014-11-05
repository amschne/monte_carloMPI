#!/bin/sh
#PBS -S /bin/sh
#PBS -N your-mpi-job
#PBS -l procs=12,mem=1gb,walltime=0:01:00
#PBS -A climate_flux
#PBS -l qos=flux
#PBS -q flux
#PBS -M amschne@umich.edu
#PBS -m abe
#PBS -j oe
#PBS -V
#
echo "I ran on:"
cat $PBS_NODEFILE
#cd ~/your_stuff

# Use mpirun to run with 12 cores
mpirun mpi4py-examples/01-hello-world
