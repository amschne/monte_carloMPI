Monte Carlo model for multiple scattering in snowpack
===============================================================================

This repository contains codes used to simulate multiple scattering (in 3D)
in idealized snowpacks. It can be used to estimate bidirectional reflectance
of a single layer snowpack for a given snow grain size. For details, please
see the article published by Schneider et al. (2019).
-------------------------------------------------------------------------------

# How to execute a model simulation
Executing the model requires an mpi4py installation and Python 2. The model has run
successfully on a computer Red Hat Enterprise Linux (v8.8) via a miniconda2
installation (see
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
    
    monte_carlo3D-run.py
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

## 5. Analyze results
Results from a model simulation are stored in a CSV plain text file. By default,
the output file is written to monte_carlo_results/sphere/ (for spheres). The text file
name follows the format

    WVL_FWHM_REFF_NPHOTON_THETA_HG.txt

where WVL and FWHM are the center wavelength and full width at half-maximum of the
distribution of the incident photon flux (micro-meters), REFF is the model snow
grain effective radius (micro-meters), NPHOTON is the number of photon packets
traced during the simulation, THETA is the zenith angle of the incident
photon flux, and "HG" indicating that the Henyey-Greenstein phase function
(van de Hulst, 1968) was used to determine scattering directions. The text file
is organized by column with the following headings:

    condition - Integer from 1 to 4 indicating the end result of a photon packet:
        1: photon packet exits top of snowpack (reflected)
        2: photon packet exits bottom of snowpack (diffuse transmittance)
        3: photon packet exits bottom of snowpack without any scattering events (direct transmittance)
        4: photon packet absorbed
    wvn[um^-1] - wavenumber corresponding with the photon packet 
    theta_n - zenith angle of photon packet's final propogation before exiting or reflected by snowpack (radians)
    phi_n - azimuthal angle of photon packet's final propogation before exiting or reflected by snowpack (radians)
    n_scat - number of scattering events encountered by a photon packet
    path_length[m] - Distance of photon packet's propogation, including multiple scattering, within snowpack
    snow_depth[m] - Effective depth of snowpack

The results of a simulation can be visualized via a polar plot of the bidirectional reflectance
factor, e.g., 

    python monte_carloMPI/post_processing.py --output_dir monte_carlo_results/sphere $(ls monte_carlo_results/sphere/*.txt | xargs)

# References
Bohren, C. F., & Huffman, D. R. (1983). Absorption and Scattering of Light by Small Particles. _Wiley_.

Schneider, A., Flanner, M., De Roo, R., and Adolph, A. (2019). Monitoring of snow surface near-infrared bidirectional reflectance factors with added light-absorbing particles, _The Cryosphere, 13_, 1753–1766, https://doi.org/10.5194/tc-13-1753-2019.

van de Hulst, H. (1968), Asymptotic fitting, a method for solving anisotropic transfer problems in thick layers, _J. Comput. Phys._, 3, 291–306, https://doi.org/10.1016/0021-9991(68)90023-5.

Yang, P., Bi, L., Baum, B. A., Liou, K.-N., Kattawar, G. W., Mishchenko, M. I., and Cole, B. (2013). Spectrally Consistent Scattering, Absorption, and Polarization Properties of Atmospheric Ice Crystals at Wavelengths from 0.2 to 100 µm, _J. Atmos. Sci., 70_, 330–347, https://doi.org/10.1175/JAS-D-12-039.1.
