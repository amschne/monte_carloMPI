#!/usr/bin/env python

""" This file (monte_carlo3D.py) is the python version of Mark Flanner's
    "monte_carlo3D.m", a matlab routine simulating photon pathways through a
    snow pack in 3 dimensions.
"""

import os

import ConfigParser
import argparse

import numpy as np
from scipy.io import netcdf
from matplotlib import pyplot as plt

class MonteCarlo(object):
    def __init__(self, **model_kwargs):
        """ valid model_kwargs:
            tau_tot = [SNOW OPTICAL DEPTH]
            imp_cnc = [MASS CONCENTRATION OF IMPURITY [mIMP/(mIMP+mICE)]]
            rho_snw = [SNOW DENSITY (kg/m3, only needed if flg_crt=1)]
            rsensor = [SENSOR RADIUS [m]]
            hsensor = [SENSOR HIEGHT ABOVE SNOW [m]]
            flg_crt = plot in optical depth space (=0) or Cartesian space (=1)?
            flg_3D = plot in 2-D (=0), 3-D (=1). or no plot (=999)?
            optics_dir = [DIRECTORY OF OPTICS FILES]
            fi_imp = [IMPURITY OPTICS FILE]
        """       
        model_args = self.get_model_args()
        
        # build dictionary of model_args
        model_args_dict = {'tau_tot' : model_args.tau_tot,
                           'imp_cnc' : model_args.imp_cnc,
                           'rho_snw' : model_args.rho_snw,
                           'rsensor' : model_args.rsensor,
                           'hsensor' : model_args.hsensor,
                           'flg_crt' : model_args.flg_crt,
                           'flg_3D' : model_args.flg_3D,
                           'optics_dir' : model_args.optics_dir,
                           'fi_imp' : model_args.fi_imp}
        
        # overwrite model_args_dict[kwarg] if specified at instantiation
        for kwarg, val in model_kwargs.items():
            model_args_dict[kwarg] = val
        
        self.tau_tot = model_args_dict['tau_tot']
        self.imp_cnc = model_args_dict['imp_cnc']
        self.rho_snw = model_args_dict['rho_snw']
        self.rsensor = model_args_dict['rsensor']
        self.hsensor = model_args_dict['hsensor']
        self.flg_crt = model_args_dict['flg_crt']
        self.flg_3D = model_args_dict['flg_3D']
        self.optics_dir = model_args_dict['optics_dir']
        self.fi_imp = model_args_dict['fi_imp']
    
    def get_optical_properties(self, wvls, rds_snw):
        """ Retrieve snow and impurity optical properties from NetCDF files,
            based on user-specified wavelengths (wvls), snow grain size
            (rds_snw), and impurity optics file (self.fi_imp)
            
            Returns ssa_ice, ext_cff_mss_ice, g, ssa_imp, ext_cff_mss_imp
        """
        # snow optics:
        fi_name = 'ice_wrn_%04d.nc' % rds_snw
        fi = os.path.join(self.optics_dir, fi_name)
        snow_optics = netcdf.netcdf_file(fi, 'r')
        
        wvl_in = snow_optics.variables['wvl']
        ssa_in = snow_optics.variables['ss_alb']
        ext_in = snow_optics.variables['ext_cff_mss']
        asm_in = snow_optics.variables['asm_prm']
        
        if type(wvls)==int or type(wvls)==float:
            idx_wvl = np.argmin(np.absolute(wvl*1e-6 - wvl_in.data))
            ssa_ice = ssa_in[idx_wvl]
            ext_cff_mss_ice = ext_in[idx_wvl]
            g = asm_in[idx_wvl]
        
        elif type(wvls)==np.ndarray:
            ssa_ice = np.empty(wvls.shape)
            ext_cff_mss_ice = np.empty(wvls.shape)
            g = np.empty(wvls.shape)     
            for i, wvl in enumerate(wvls):
                idx_wvl = np.argmin(np.absolute(wvl*1e-6 - wvl_in.data))
                ssa_ice[i] = ssa_in[idx_wvl]
                ext_cff_mss_ice[i] = ext_in[idx_wvl]
                g[i] = asm_in[idx_wvl]
        
        snow_optics.close()

        # impurity optics:
        fi_imp = os.path.join(self.optics_dir, self.fi_imp)
        impurity_optics = netcdf.netcdf_file(fi_imp, 'r')
        
        wvl_in_imp = impurity_optics.variables['wvl']
        ssa_in_imp = impurity_optics.variables['ss_alb']
        ext_in_imp = impurity_optics.variables['ext_cff_mss']
        
        if type(wvls)==int or type(wvls)==float:
            idx_wvl = np.argmin(np.absolute(wvl*1e-6 - wvl_in_imp.data))
            ssa_imp = ssa_in_imp[idx_wvl]
            ext_cff_mss_imp = ext_in_imp[idx_wvl]
        elif type(wvls)==np.ndarray:
            ssa_imp = np.empty(wvls.shape)
            ext_cff_mss_imp = np.empty(wvls.shape)
            for i, wvl in enumerate(wvls):
                idx_wvl = np.argmin(np.absolute(wvl*1e-6 - wvl_in_imp.data))
                ssa_imp[i] = ssa_in_imp[idx_wvl]
                ext_cff_mss_imp[i] = ext_in_imp[idx_wvl]
   
        impurity_optics.close()
    
        return(ssa_ice, ext_cff_mss_ice, g, ssa_imp, ext_cff_mss_imp)
    
    def Henyey_Greenstein(self):
        """ Henyey-Greenstein scattering phase function
        """
        costheta_p = self.costheta_p
        g = np.matrix(self.g).T
        g_2 = np.multiply(g,g) # compute g^2
        HG_num = 1 - g_2
        HG_den = np.power(1 + g_2 - 2*g*costheta_p, 3./2.)
        p_HG = HG_num / HG_den
        
        return p_HG
    
    def run(self, n_photon, wvl0, half_width, rds_snw, test=False):
        """ Run the Monte Carlo model given a normal distribution of
            wavelengths [um].  This better simulates what NERD does with
            non-monochromatic LEDs.
            
            ALL VALUES IN MICRONS
        """
        # Convert half_width to standard deviation
        scale = (2 * half_width) / 2.355
        
        # Generate random array of photon wavelengths, rounded to nearest nm
        wvls = np.around(np.random.normal(loc=wvl0, scale=scale,
                                          size=(n_photon)), decimals=3)
        (ssa_ice,
         ext_cff_mss_ice,
         g, 
         ssa_imp,
         ext_cff_mss_imp) = self.get_optical_properties(wvls, rds_snw)
         
        if test:
            try:
                ssa_ice = self.ssa_ice
            except AttributeError:
                pass
            try:
                ext_cff_mss_ice = self.ext_cff_mss_ice
            except AttributeError:
                pass
            try:
                g = self.g
            except AttributeError:
                pass
            try:
                ssa_imp = self.ssa_imp
            except AttributeError:
                pass
            try:
                ext_cff_mss_imp = self.ext_cff_mss_imp
            except AttributeError:
                pass
                
        imp_cnc = self.imp_cnc
        # combined mass extinction cross-section of ice+impurity system (m2/kg)
        ext_cff_mss = ext_cff_mss_ice*(1-imp_cnc) + ext_cff_mss_imp*imp_cnc
        
        # calculate probability that extinction event is caused by impurity
        P_ext_imp = (imp_cnc*ext_cff_mss_imp) / (imp_cnc*ext_cff_mss_imp + 
                                                 (1-imp_cnc)*ext_cff_mss_ice)

        # cos(theta) array over which to compute function
        costheta_p = np.arange(-1.000, 1.001, 0.001)
        self.costheta_p = costheta_p
        
        self.g = g
        p = self.Henyey_Greenstein()
        self.p = p
    
    def plot_phase_function(self):
        """ plot phase function versus cos(theta)
        """
        plt.contourf(self.costheta_p, self.g, self.p)
    
    def monte_carlo3D(self, n_photon, wvl, rds_snw):
        """ Translated from matlab to python by Adam Schneider
        
            Returns albedo and fraction of incident photons reaching sensor
        """
            
    
    def get_model_args(self):    
        """ Specify model kwargs at run time or get values from config.ini
        """  
        config = ConfigParser.SafeConfigParser()
        data_dir = os.path.join(os.getcwd(), 'data')
        config_file = os.path.join(data_dir, 'config.ini')
        config.read(config_file)
        
        # model parameters
        section_name = 'model parameters'
        tau_tot = config.getfloat(section_name, 'tau_tot')
        imp_cnc = config.getfloat(section_name, 'imp_cnc')
        rho_snw = config.getfloat(section_name, 'rho_snw')
        rsensor = config.getfloat(section_name, 'rsensor')
        hsensor = config.getfloat(section_name, 'hsensor')
        
        # plot options
        section_name = 'plot options'
        flg_crt = config.getint(section_name, 'flg_crt')
        flg_3D = config.getint(section_name, 'flg_3D')

        # data
        section_name = 'data'
        optics_dir = config.get(section_name, 'optics_dir')
        fi_imp = config.get(section_name, 'fi_imp')

        # run time args
        parser = argparse.ArgumentParser(description='[DESCRIPTION]')
        parser.add_argument('--tau_tot', type=float, default=tau_tot,
                            help='snow optical depth')
        parser.add_argument('--imp_cnc', type=float, default=imp_cnc,
                            help='mass concentration of impurity '
                                 '[mIMP/(mIMP+mICE)]')
        parser.add_argument('--rho_snw', type=float, default=rho_snw,
                            help='snow density (kg/m3, only needed if '
                                 'flg_crt=1)')
        parser.add_argument('--rsensor', type=float, default=rsensor,
                            help='sensor radius [m]') 
        parser.add_argument('--hsensor', type=float, default=hsensor,
                            help='sensor height above snow [m]')
        parser.add_argument('--flg_crt', type=int, default=flg_crt,
                            help='plot in optical depth space (=0) or '
                                 'Cartesian space (=1)?')
        parser.add_argument('--flg_3D', type=int, default=flg_3D,
                            help='plot in 2-D (=0), 3-D (=1). '
                                 'or no plot (=999)?')
        parser.add_argument('--optics_dir', type=str, default=optics_dir, 
                            help='directory of optics files')
        parser.add_argument('--fi_imp', type=str, default=fi_imp)
        
        args = parser.parse_args()
        
        return args

def test(n_photon = 50000, wvl=1.3, half_width=0.085, rds_snw=100):
    """ Test case for comparison with Wang et al (1995) Table 1, and van de
        Hulst (1980).  Albedo should be ~0.09739.  Total transmittance
        (diffuse+direct) should be ~0.66096.
    """
    #n_photon = 50000
    tau_tot = 2.0
    ssa_ice = 0.9
    g = 0.75
    imp_cnc = 0
    
    test_case = MonteCarlo(tau_tot=tau_tot, imp_cnc=imp_cnc)
    test_case.ssa_ice = ssa_ice
    test_case.g = g
    
    test_case.run(n_photon, wvl, half_width, rds_snw, test=True)

def test_and_debug(n_photon=100, wvl=1.3, half_width=0.085, rds_snw=100):
    """ manually specify optical properties for test cases and debugging:
    """
    # mass extinction cross-section of ice grains (m2/kg at 500nm)
    ext_cff_mss_ice = 6.6 # typical for re=250um
    
    # single-scatter albedo of ice grains (at 500nm)
    ssa_ice = 0.999989859099 # typical for re=250um
    
    # scattering asymmetry parameter for ice grains
    g = 0.89 # typical for re=250
    
    # mass extinction cross-section of black carbon (m2/kg at 500nm)
    ext_cff_mss_imp = 12000
    
    # single-scatter albedo of black carbon (500nm)
    ssa_imp = 0.30
    
    test_case = MonteCarlo()
    test_case.ext_cff_mss_ice = ext_cff_mss_ice
    test_case.ssa_ice = ssa_ice
    test_case.g = g
    test_case.ext_cff_mss_imp = ext_cff_mss_imp
    test_case.ssa_imp = ssa_imp
    
    test_case.run(n_photon, wvl, half_width, rds_snw, test=True)   

def run():
    """ USER INPUT
    """
    # set number of photons
    n_photon = 100
    
    # wavelength [um]
    wvl = 1.3
    #wvl = 1.55
    #wvl = 0.5
    
    # half width [um]
    half_width = 0.085
    
    # snow effective grain size [um]
    rds_snw = 100.
    
    # snow optical depth
    tau_tot = 1000.0
    
    # mass concentration of impurity [mIMP/(mIMP+mICE)]
    imp_cnc=0e-9
    
    # plot in optical depth space (=0) or Cartesian space (=1)?
    flg_crt = 1
    
    # plot in 2-D (=0), 3-D (=1). or no plot (=999)?
    flg_3D = 0

    # snow density (kg/m3, only needed if flg_crt=1)
    rho_snw = 200.

    # directory of optics files
    #optics_dir = '/data/flanner/mie/snicar'
    #optics_dir = '/home/amaschne/Projects/monte_carloMPI/data/snicar'
    optics_dir = '/Users/amschne/Documents/research/monte_carloMPI/data/snicar'
    
    # specification for nadir looking sensor
    rsensor = 0.05 # sensor radius [m]
    hsensor = 0.1 # sensor height above snow [m]
    
    # impurity optics file
    fi_imp = 'mie_sot_ChC90_dns_1317.nc'
    
    """ END USER INPUT
    """
    monte_carlo_model = MonteCarlo(tau_tot=tau_tot, imp_cnc=imp_cnc,
                                   flg_crt=flg_crt, flg_3D=flg_3D,
                                   rho_snw=rho_snw, optics_dir=optics_dir,
                                   rsensor=rsensor, hsensor=hsensor,
                                   fi_imp=fi_imp)
                                   
    monte_carlo_model.run(n_photon, wvl, half_width, rds_snw)
    monte_carlo_model.plot_phase_function()

def main():
    run()

if __name__=='__main__':
    main()