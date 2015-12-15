### BEGIN USER INPUT ###

# SET FILEPATHS
export MONTE_CARLO_ROOT=/home/$USER/monte_carloMPI
export DATA=/scratch/climate_flux/$USER

# SET CASE NAME
export CASE=AGU2015

### END USER INPUT ###

export CASEROOT=/home/$USER/$CASE

# CREATE OUTPUT DIR
#mkdir $DATA/$CASE

# CREATE CASEROOT DIR, COPY RUN SCRIPTS, CONFIG FILE, POST PROCESSING SCRIPT
#mkdir $CASEROOT
#cp $MONTE_CARLO_ROOT/monte_carlo3D-run.py $CASEROOT
#cp $MONTE_CARLO_ROOT/monte_carloMPI.pbs $CASEROOT
#cp $MONTE_CARLO_ROOT/config.ini $CASEROOT
#cp $MONTE_CARLO_ROOT/post_processing.py $CASEROOT
#cp $MONTE_CARLO_ROOT/polar_demo.py $CASEROOT

# COPY THIS SCRIPT TO CASEROOT FOR FUTURE REFERENCE
#cp $MONTE_CARLO_ROOT/model_setup.sh $CASEROOT
#cp $MONTE_CARLO_ROOT/set_env_vars.sh $CASEROOT

#cd $CASEROOT

# LOAD MODULES
module unload openmpi
module load gcc/4.4.6
module load openmpi/1.6.0/gcc/4.4.6
module load python
module load mpi4py/mpi-1.6.0/gcc-4.4.6/1.3

# AUGMENT PYTHONPATH
export PYTHONPATH=$MONTE_CARLO_ROOT/monte_carloMPI

echo ' '
echo 'CASEROOT (cwd) setup'
echo '--------------------'
echo 'Edit config.ini for basic configuration options'
echo 'In config.ini, be sure to set output_dir=$DATA/$CASE'
echo 'Edit monte_carlo3D-run.py to setup run parameters'
echo 'Finally, $ qsub monte_carloMPI.pbs # to submit job'
echo ' '