#!/usr/bin/env python

""" This file (monte_carlo3D.py) is the python version of Mark Flanner's
    "monte_carlo3D.m", a matlab routine simulating photon pathways through a
    snow pack in 3 dimensions.
"""

import os

import ConfigParser
import argparse

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
            data_dir = [DIRECTORY OF OPTICS FILES]
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
                           'data_dir' : model_args.data_dir}
        
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
        self.data_dir = model_args_dict['data_dir']
    
    def monte_carlo3D(n_photon, wvl, rds_snw):
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
        optics_dir = config.get(section_name, 'dir')

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
        parser.add_argument('--optics_dir', type=str, default=data_dir, 
                            help='directory of optics files')
        
        args = parser.parse_args()
        
        return args

def run():
    """ USER INPUT
    """
    # set number of photons
    n_photon = 100
    
    # wavelength [um]
    #wvl = 1.3
    #wvl = 1.55
    wvl = 0.5
    
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
    optics_dir = '/home/amaschne/mie/snicar'
    
    # specification for nadir looking sensor
    rsensor = 0.05 # sensor radius [m]
    hsensor = 0.1 # sensor height above snow [m]
    
    """ END USER INPUT
    """
    monte_carlo_model = MonteCarlo(tau_tot=tau_tot, imp_cnc=imp_cnc,
                                   flg_crt=flg_crt, flg_3D=flg_3D,
                                   rho_snw=rho_snw, optics_dir=optics_dir,
                                   rsensor=rsensor, hsensor=hsensor)
    monte_carlo_model.monte_carlo3D(n_photon, wvl, rds_snw)

def main():
    run()

if __name__=='__main__':
    main()
