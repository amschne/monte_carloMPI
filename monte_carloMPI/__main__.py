#!/usr/bin/env python

import numpy as np
import monte_carlo3D

DEBUG = False

def single_grain_size(n_photon, wvl, half_width, rds_snw):
    monte_carlo_run = monte_carlo3D.MonteCarlo()
    monte_carlo_run.run(n_photon, wvl, half_width, rds_snw, debug=DEBUG)
    #monte_carlo_run.plot_phase_function()
    
def multiple_grain_sizes(n_photon, wvl, half_width, rds_snw):
    monte_carlo_run = monte_carlo3D.MonteCarlo()
    R = np.empty(np.shape(rds_snw))
    for i, rds in enumerate(rds_snw):
        R[i] = monte_carlo_run.run(n_photon, wvl, half_width, rds, debug=DEBUG)
        #monte_carlo_run.plot_phase_function()
        
    return R

def run():    
    """ USER INPUT
    """
    # set number of photons
    n_photon = 1000
    
    # wavelength [um]
    wvl = 1.3
    #wvl = 1.55
    #wvl = 0.5
    
    # half width [um]
    half_width = 0.085
    
    # snow effective grain size [um]
    if True: # single grain size
        rds_snw = 100.
        single_grain_size(n_photon, wvl, half_width, rds_snw)
    
    if False: # multiple grain sizes
        rds_snw = np.arange(100., 120., 10)
        multiple_grain_sizes(n_photon, wvl, half_width, rds_snw)
        
    """ END USER INPUT
    """
def main():
   run()

if __name__=='__main__':
    main()