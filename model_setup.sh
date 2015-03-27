# SET FILEPATHS
export MONTE_CARLO_ROOT=/home/$USER/monte_carloMPI
export DATA=/scratch/climate_flux/$USER

# SET CASE NAME
export CASE=compare_with_mark
export CASEROOT=/home/$USER/$CASE

# CREATE OUTPUT DIR
mkdir $DATA/$CASE

# CREATE CASEROOT DIR, COPY RUN SCRIPTS, CONFIG FILE, POST PROCESSING SCRIPT
mkdir $CASEROOT
cp $MONTE_CARLO_ROOT/monte_carlo3D-run.py $CASEROOT
cp $MONTE_CARLO_ROOT/monte_carloMPI.pbs $CASEROOT
cp $MONTE_CARLO_ROOT/config.ini $CASEROOT
cp $MONTE_CARLO_ROOT/post_processing.py $CASEROOT
cp $MONTE_CARLO_ROOT/polar_demo.py $CASEROOT

# COPY THIS SCRIPT TO CASEROOT FOR FUTURE REFERENCE
cp $MONTE_CARLO_ROOT/model_setup.sh $CASEROOT

cd $CASEROOT

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