Monte Carlo model for multiple scattering in snowpack
===============================================================================

This repository contains codes used to simulate multiple scattering (in 3D)
in idealized snowpacks. It can be used to estimate bidirectional reflectance
of a single layer snowpack for a given snow grain size. For details, please
see the article published by Schneider et al. (2019).
-------------------------------------------------------------------------------

# How to execute a model simulation
Executing the model requires an mpi4py installation. The model has run
successfully on computers running macOS (v13.4) and Red Hat Enterprise
Linux (v8.8) via a conda installation (see
https://docs.conda.io/en/latest/miniconda.html to install miniconda).

## 0. Download this repository
Note that the following command pertains to the development version of the
project stored on GitHub. If you already obtained an archive of this repository
(e.g., via Zotero), move to step 1.
    
    git clone  https://github.com/amschne/monte_carloMPI.git; cd monte_carloMPI
## 1. Import the conda environment
    conda env create -n ENVNAME --file environment.yml
## 2. Unpack model input data, originally generated with the Mie solver of Bohren and Huffman (1983)
    tar -xzvf Bohren_and_Huffman_1983_Mie_solver_data.tar.gz
## 3. Configure the model simulation
Model simulation parameters can be specified by editing the driver script
    vim monte_carlo3D-run.py
In the driver script, n_photon sets the number of photon (packets). Although
the default is set to 10 thousand photon packets, Schneider et al. (2019) found
that at least 1 million photon packets are needed to acheive stochastic convergence.
Because tracing 1 million photon packets is computationally time consuming,
the default of 10 thousand allows users to more easily test their installation.

The model relies on data that contain ice crystal single-scattering properties (SSPs).
To execute a model simulation that approximates snowpack as a medium of (small)
spherical ice particles suspended in air, simply follow the remaining steps
provided in this README. Although we have archived the SSP data for spheres,
SSP data for non-spherical particles were developed by Yang et al. (2013) and are
not available from this archive.

## 4. Execute the model simulation
    mpirun -np NUMBER_OF_TASKS --oversubscribe monte_carlo3D-run.py

# References
