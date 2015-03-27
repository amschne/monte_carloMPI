#!/bin/sh
#PBS -S /bin/sh
##PBS -N your-mpi-job
#PBS -l procs=12,pmem=1gb,walltime=1:00:00
#PBS -A climate_flux
#PBS -l qos=flux
#PBS -q flux
#PBS -M amschne@umich.edu
#PBS -m abe
#PBS -j oe
#PBS -V

module unload openmpi
module load gcc/4.4.6
module load openmpi/1.6.0/gcc/4.4.6
module load python
module load mpi4py/mpi-1.6.0/gcc-4.4.6/1.3

echo "I ran on:"
cat $PBS_NODEFILE
cd $CASEROOT

# Use mpirun to run with n cores (#PBS -l procs=...)
mpirun -N $CASE monte_carlo3D-run.py