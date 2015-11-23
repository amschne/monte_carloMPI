#!/usr/bin/env python

import numpy as np
import monte_carlo3D

DEBUG = False
LAMBERTIAN_SURFACE = False # True will simply simulate a
                           # Lambertian surface
LAMBERTIAN_BOTTOM = True # True will add Lambertian bottom boundary with reflectance
                         # set below
LAMBERTIAN_REFLECTANCE = 1e-9 # Set reflectance of Lambertian surface OR
                              # reflectance of underlying surface beneath snow 

def run():    
    """ USER INPUT
    """
    # set number of photons
    n_photon = 1000000
    
    # set initial incidence zenith angle (degrees)
    theta_0 = 0.
    
    # set initial Stokes parameters
    I = 1
    Q = 0
    U = 0
    V = 0
    
    stokes_params = np.array([I, Q, U, V])
    
    # select one of the following snow grain shape habits:
    #shape = 'sphere' # default
    shape = 'solid hexagonal column'
    #shape = 'hexagonal plate'
    #shape = 'hollow hexagonal column'
    #shape = 'droxtal'
    #shape = 'hollow bullet rosette'
    #shape = 'solid bullet rosette'
    #shape = '8-element column aggregate'
    #shape = '5-element plate aggregate'
    #shape = '10-element plate aggregate'
    
    # select degree of surface roughness:
    #roughness = 'smooth' # default
    #roughness = 'moderatley rough'
    roughness = 'severely rough'
    
    # wavelength [um]
    #wvl = 1.3
    #wvl = 1.55
    #wvl = 0.5
    
    # half width [um]
    #half_width = 0.085
    #half_width = 0.3
    
    if True:  # use for runs of multiple wavelengths
        #wvls = np.arange(0.305, 3., 0.1)
        wvls = [1.3, 1.55]
        half_widths = [0.085, 0.130]
        
        for i, wvl in enumerate(wvls):
            half_width = half_widths[i]
            # snow effective grain size [um]
            if False: # single grain size
                rds_snw = 10.
                single_grain_size(n_photon, wvl, half_width, rds_snw,
                                  theta_0=theta_0, stokes_params=stokes_params,
                                  shape=shape, roughness=roughness)
            
            if True: # multiple grain sizes
                #rds_snw = np.arange(10., 1000., 100.)
                rds_snw = np.array([50, 100, 250, 500, 1000])
                multiple_grain_sizes(n_photon, wvl, half_width, rds_snw,
                                     theta_0=theta_0,
                                     stokes_params=stokes_params,
                                     shape=shape, roughness=roughness)
    
    if False:  # use for run of single wavelength
        # snow effective grain size [um]
        if True: # single grain size
            rds_snw = 100.
            single_grain_size(n_photon, wvl, half_width, rds_snw,
                              theta_0=theta_0, stokes_params=stokes_params,
                              shape=shape, roughness=roughness)
    
        if False: # multiple grain sizes
            rds_snw = np.array([50, 100, 200, 500, 1000])
            multiple_grain_sizes(n_photon, wvl, half_width, rds_snw,
                                 theta_0=theta_0, stokes_params=stokes_params,
                                 shape=shape, roughness=roughness)
        
    """ END USER INPUT
    """
    
def single_grain_size(n_photon, wvl, half_width, rds_snw, theta_0=0.,
                      stokes_params=np.array([1,0,0,0]),
                      shape='sphere', roughness='smooth'):
    monte_carlo_run = monte_carlo3D.MonteCarlo()
    monte_carlo_run.run(n_photon, wvl, half_width, rds_snw, theta_0=theta_0,
                        stokes_params=stokes_params, shape=shape,
                        roughness=roughness, debug=DEBUG,
                        Lambertian_surface=LAMBERTIAN_SURFACE,
                        Lambertian_bottom=LAMBERTIAN_BOTTOM,
                        Lambertian_reflectance=LAMBERTIAN_REFLECTANCE)
    
def multiple_grain_sizes(n_photon, wvl, half_width, rds_snw, theta_0=0.,
                         stokes_params=np.array([1,0,0,0]), shape='sphere',
                         roughness='smooth'):
    monte_carlo_run = monte_carlo3D.MonteCarlo()
    for i, rds in enumerate(rds_snw):
        monte_carlo_run.run(n_photon, wvl, half_width, rds, theta_0=theta_0,
                            stokes_params=stokes_params, shape=shape,
                            roughness=roughness, debug=DEBUG,
                            Lambertian_surface=LAMBERTIAN_SURFACE,
                            Lambertian_bottom=LAMBERTIAN_BOTTOM,
                            Lambertian_reflectance=LAMBERTIAN_REFLECTANCE)
        
def main():
   run()

if __name__=='__main__':
    main()
