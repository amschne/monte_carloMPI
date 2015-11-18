#!/usr/bin/env python

""" This file (monte_carlo3D.py) is the python version of Mark Flanner's
    "monte_carlo3D.m", a matlab routine simulating photon pathways through a
    snow pack in 3 dimensions.
"""
import os
import sys

import ConfigParser
import argparse

import numpy as np
from scipy import interpolate
from scipy.io import netcdf
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from parallelize import Parallel

#import ipdb
import time
#from memory_profiler import profile

TWO_PIE = 2*np.pi
FOUR_PIE = 4*np.pi

def timefunc(f):
    """ Used for timing function calls

        To use, simply place "@timefunc" above the function you wish to time
    """
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, 'took', end - start, 'time'
        return result
    return f_timer

class MonteCarlo(object):
    def __init__(self, **model_kwargs):
        """ valid model_kwargs:
            tau_tot = [SNOW OPTICAL DEPTH]
            imp_cnc = [MASS CONCENTRATION OF IMPURITY [mIMP/(mIMP+mICE)]]
            rho_snw = [SNOW DENSITY (kg/m3, only needed if flg_crt=1)]
            rho_ice = [ICE DENSITY (kg/m3)]
            rsensor = [SENSOR RADIUS [m]]
            hsensor = [SENSOR HIEGHT ABOVE SNOW [m]]
            flg_crt = plot in optical depth space (=0) or Cartesian space (=1)?
            flg_3D = plot in 2-D (=0), 3-D (=1). or no plot (=999)?
            output_dir = [DIRECTORY TO WRITE OUTPUT DATA TO]
            optics_dir = [DIRECTORY OF OPTICS FILES]
            fi_imp = [IMPURITY OPTICS FILE]
            HG = [True] to force Henyey-Greenstein phase function for aspherical
                 parcicles
            phase_functions = [True] to plot scattering phase functions
                                  instead of running model
        """       
        model_args = self.get_model_args()
        
        # build dictionary of model_args
        model_args_dict = {'tau_tot' : model_args.tau_tot,
                           'imp_cnc' : model_args.imp_cnc,
                           'rho_snw' : model_args.rho_snw,
                           'rho_ice' : model_args.rho_ice,
                           'rsensor' : model_args.rsensor,
                           'hsensor' : model_args.hsensor,
                           'flg_crt' : model_args.flg_crt,
                           'flg_3D' : model_args.flg_3D,
                           'output_dir' : model_args.output_dir,
                           'optics_dir' : model_args.optics_dir,
                           'fi_imp' : model_args.fi_imp,
                           'HG': model_args.HG,
                           'phase_functions': model_args.phase_functions}
        
        # overwrite model_args_dict[kwarg] if specified at instantiation
        for kwarg, val in model_kwargs.items():
            model_args_dict[kwarg] = val
        
        self.tau_tot = model_args_dict['tau_tot']
        self.imp_cnc = model_args_dict['imp_cnc']
        self.rho_snw = model_args_dict['rho_snw']
        self.rho_ice = model_args_dict['rho_ice']
        self.rsensor = model_args_dict['rsensor']
        self.hsensor = model_args_dict['hsensor']
        self.flg_crt = model_args_dict['flg_crt']
        self.flg_3D = model_args_dict['flg_3D']
        self.output_dir = model_args_dict['output_dir']
        self.optics_dir = model_args_dict['optics_dir']
        self.fi_imp = model_args_dict['fi_imp']
        self.HG = model_args_dict['HG']
        self.phase_functions = model_args_dict['phase_functions']
    
    def setup_output(self, n_photon, wvl0, half_width, rds_snw):
        """ Create output dir for writing data to
        
            Returns output_file path
        """
        self.output_dir
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
            
        run_name = '%s_%s_%s_%s.txt' % (wvl0, half_width, rds_snw, n_photon)
        output_file = os.path.join(self.output_dir, run_name)
        i = 0
        while os.path.isfile(output_file):
            i += 1
            run_name = '%s_%s_%s_%s_%d.txt' % (wvl0, half_width, rds_snw,
                                               n_photon, i)
            output_file = os.path.join(self.output_dir, run_name)
            
        return output_file
    
    #@timefunc
    def get_aspherical_SSPs(self, wvls, rds_snw):
        """ Retrieve single scattering properties (SSPs) for aspherical ice
            particle shapes, based on input wavelengths (wvls) and snow and
            effective radii (rds_snw).
            
            Returns ssa_ice, ext_cff_mss_ice, g
        """
        if self.far_IR:
            wvl_dir = '16.4-99.0'
        else:
            wvl_dir = '0.2-15.25'
            
        if self.shape == 'solid hexagonal column':
            shape_dir = 'solid_column'
        elif self.shape == 'hexagonal plate':
            shape_dir = 'plate'
        elif self.shape == 'hollow hexagonal column':
            shape_dir = 'hollow_column'
        elif self.shape == 'droxtal':
            shape_dir = 'droxtal'
        elif self.shape == 'hollow bullet rosette':
            shape_dir = 'hollow_bullet_rosette'
        elif self.shape == 'solid bullet rosette':
            shape_dir = 'solid_bullet_rosette'
        elif self.shape == '8-element column aggregate':
            shape_dir = 'column_8elements'
        elif self.shape == '5-element plate aggregate':
            shape_dir = 'plate_5elements'
        elif self.shape =='10-element plate aggregate':
            shape_dir = 'plate_10elements'
            
        if self.roughness == 'smooth':
            roughness_dir = 'Rough000'
        elif self.roughness == 'moderatley rough':
            roughness_dir = 'Rough003'
        elif self.roughness == 'severely rough':
            roughness_dir = 'Rough050'
            
        fi_name = 'isca.dat'
        aspherical_particle_dir = os.path.join(self.optics_dir, 'ice_optics', 
                                               wvl_dir, shape_dir,roughness_dir)
        fi = os.path.join(aspherical_particle_dir, fi_name)
        
        aspherical_optics = open(fi, 'r')
        lines = aspherical_optics.readlines()
        wvl_in = list() # wavelength (um)
        max_dim_in = list() # maximum dimension of particle size (um)
        particle_volume_in = list() # volume of particle (um^3)
        G_in = list() # projected area (um^2)
        Q_ext_in = list() # extinction efficiency
        ssa_in = list() # single-scattering albedo
        asm_in = list() # asymmetry factor
        for i, line in enumerate(lines):
            wvl_in.append(float(line.split()[0]))
            max_dim_in.append(float(line.split()[1]))
            particle_volume_in.append(float(line.split()[2]))
            G_in.append(float(line.split()[3]))
            Q_ext_in.append(float(line.split()[4]))
            ssa_in.append(float(line.split()[5]))
            asm_in.append(float(line.split()[6]))
        
        # Scattering phase matrix elements    
        if not self.HG:
            P11_filename = 'P11.dat'
            P12_filename = 'P12.dat'
            P22_filename = 'P22.dat'
            P33_filename = 'P33.dat'
            P43_filename = 'P43.dat'
            P44_filename = 'P44.dat'
            
            P11_fi = os.path.join(aspherical_particle_dir, P11_filename)
            P12_fi = os.path.join(aspherical_particle_dir, P12_filename)
            P22_fi = os.path.join(aspherical_particle_dir, P22_filename)
            P33_fi = os.path.join(aspherical_particle_dir, P33_filename)
            P43_fi = os.path.join(aspherical_particle_dir, P43_filename)
            P44_fi = os.path.join(aspherical_particle_dir, P44_filename)
            
            P11_dat = open(P11_fi, 'r')
            P12_dat = open(P12_fi, 'r')
            P22_dat = open(P22_fi, 'r')
            P33_dat = open(P33_fi, 'r')
            P43_dat = open(P43_fi, 'r')
            P44_dat = open(P44_fi, 'r')
            
            P11_line0 = P11_dat.readline().split()
            P11_lines = P11_dat.readlines()
            P12_line0 = P12_dat.readline().split()
            P12_lines = P12_dat.readlines()
            P22_line0 = P22_dat.readline().split()
            P22_lines = P22_dat.readlines()
            P33_line0 = P33_dat.readline().split()
            P33_lines = P33_dat.readlines()
            P43_line0 = P43_dat.readline().split()
            P43_lines = P43_dat.readlines()
            P44_line0 = P44_dat.readline().split()
            P44_lines = P44_dat.readlines()
            
        # Calculate effective radius
        RE = (3./4.) * (np.array(particle_volume_in) / np.array(G_in)) # um
        
        # get index with smallest abs(rds_snw - RE)
        idx_RE = np.argsort(np.absolute(rds_snw - RE))[0]
        valid_idxs = np.where(RE == RE[idx_RE])
        self.snow_effective_radius = RE[idx_RE]
        
        # screen data for relevent snow radii
        wvl_in = np.array(wvl_in)[valid_idxs]
        max_dim_in = np.array(max_dim_in)[valid_idxs]
        particle_volume_in = np.array(particle_volume_in)[valid_idxs]
        G_in = np.array(G_in)[valid_idxs]
        Q_ext_in = np.array(Q_ext_in)[valid_idxs]
        ssa_in = np.array(ssa_in)[valid_idxs]
        asm_in = np.array(asm_in)[valid_idxs]
        
        if not self.HG:
            P11_lines = np.array(P11_lines)[valid_idxs]
            P12_lines = np.array(P12_lines)[valid_idxs]
            P22_lines = np.array(P22_lines)[valid_idxs]
            P33_lines = np.array(P33_lines)[valid_idxs]
            P43_lines = np.array(P43_lines)[valid_idxs]
            P44_lines = np.array(P44_lines)[valid_idxs]
            
            P11 = dict()
            P12 = dict()
            P22 = dict()
            P33 = dict()
            P43 = dict()
            P44 = dict()
            
            P12_norm = dict()
            P22_norm = dict()
            P33_norm = dict()
            P43_norm = dict()
            P44_norm = dict()
            
            theta_P11_deg = np.empty(len(P11_line0))
            theta_P12_deg = np.empty(len(P12_line0))
            theta_P22_deg = np.empty(len(P22_line0))
            theta_P33_deg = np.empty(len(P33_line0))
            theta_P43_deg = np.empty(len(P43_line0))
            theta_P44_deg = np.empty(len(P44_line0))
            
            # set up container for appropriate wvls
            self.wvls = np.empty(wvls.shape)
            wvl0_exists = False
            
        # fetch data for relevent wavelengths
        ssa_ice = np.empty(wvls.shape)
        ext_cff_mss_ice = np.empty(wvls.shape)
        g = np.empty(wvls.shape)
        
        sorted_wvls = np.sort(wvls)
        last_wvl = None 
        for i, wvl in enumerate(sorted_wvls):
            if wvl != last_wvl:
                last_wvl = wvl
                working_set_idxs = np.where(wvls==wvl)
                
                if wvl==self.wvl0:
                    self.wvl0_idx = working_set_idxs[0][0] 
                
                # get indicies with smallest abs(wvl - wvl_in)
                idx_wvl = np.argsort(np.absolute(wvl - wvl_in))
                nearest_wvls = wvl_in[idx_wvl[:2]]
            
                if self.HG: # interpolate across wvls
                    # ssa_ice
                    nearest_ssa_ice = ssa_in[idx_wvl[:2]]
                    try:
                        if nearest_wvls[0] < nearest_wvls[1]:
                            ssa_ice_interp = interpolate.interp1d(nearest_wvls,
                                                                  nearest_ssa_ice)
                            ssa_ice[working_set_idxs] = ssa_ice_interp(wvl)
                        else:
                            ssa_ice_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                                  nearest_ssa_ice[::-1])
                            ssa_ice[working_set_idxs] = ssa_ice_interp(wvl)
                    except ValueError:
                        ssa_ice[working_set_idxs] = nearest_ssa_ice[0]
                        sys.stderr.write('error: exception raised while interpolating '
                                         'ssa_ice, using nearest value instead\n')
                                 
                    # ext_cff_mss_ice
                    nearest_Q_ext = Q_ext_in[idx_wvl[:2]]
                    try:
                        if nearest_wvls[0] < nearest_wvls[1]:
                            Q_ext_interp = interpolate.interp1d(nearest_wvls, 
                                                                nearest_Q_ext)
                            Q_ext = Q_ext_interp(wvl)
                        else:
                            Q_ext_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                                nearest_Q_ext[::-1])
                            Q_ext = Q_ext_interp(wvl)
                    except ValueError:
                        Q_ext = nearest_Q_ext[0]
                        sys.stderr.write('error: exception raised while interpolating '
                                         'Q_ext, using nearest value instead\n')
                                 
                    ext_cff_mss_ice[working_set_idxs] = ((1e6*G_in[idx_wvl[0]]*Q_ext) /
                                        (self.rho_ice * particle_volume_in[idx_wvl[0]]))
            
                    # g
                    nearest_g = asm_in[idx_wvl[:2]]
                    try:
                        if nearest_wvls[0] < nearest_wvls[1]:
                            g_interp = interpolate.interp1d(nearest_wvls, nearest_g)
                            g[working_set_idxs] = g_interp(wvl)
                        else:
                            g_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                            nearest_g[::-1])
                            g[working_set_idxs] = g_interp(wvl)
                    except ValueError:
                        g[working_set_idxs] = nearest_g[0]
                        sys.stderr.write('error: exception raised while interpolating '
                                         'g, using nearest value instead\n')
                else: # NO interpolation across wvls
                    # wvl
                    key = nearest_wvls[0]
                    self.wvls[working_set_idxs] = key
                    if self.wvl0 == key:
                        wvl0_exists = True
                    
                    # ssa_ice
                    ssa_ice[working_set_idxs] = ssa_in[idx_wvl[0]]
                                 
                    # ext_cff_mss_ice
                    Q_ext = Q_ext_in[idx_wvl[0]]
                                                 
                    ext_cff_mss_ice[working_set_idxs] = ((1e6*G_in[idx_wvl[0]]*Q_ext) /
                                        (self.rho_ice * particle_volume_in[idx_wvl[0]]))
            
                    # g
                    g[working_set_idxs] = asm_in[idx_wvl[0]]
                
                    # Scattering phase matrix elements
                    # P11
                    P11[key] = np.empty(len(P11_line0))
                    for j, theta in enumerate(P11_line0):
                        if i==0:
                            theta_P11_deg[j] = float(theta)
                        P11[key][j] = float(P11_lines[idx_wvl[0]].split()[j])
                    
                    # P12
                    P12_norm[key] = np.empty(len(P12_line0))
                    for j, theta in enumerate(P12_line0):
                        if i==0:
                            theta_P12_deg[j] = float(theta)
                        P12_norm[key][j] = float(P12_lines[idx_wvl[0]].split()[j])
                    
                    P12[key] = P12_norm[key] * P11[key]
                    
                    # P22
                    P22_norm[key] = np.empty(len(P22_line0))
                    for j, theta in enumerate(P22_line0):
                        if i==0:
                            theta_P22_deg[j] = float(theta)
                        P22_norm[key][j] = float(P22_lines[idx_wvl[0]].split()[j])
                    
                    P22[key] = P22_norm[key] * P11[key]
                    
                    # P33
                    P33_norm[key] = np.empty(len(P33_line0))
                    for j, theta in enumerate(P33_line0):
                        if i==0:
                            theta_P33_deg[j] = float(theta)
                        P33_norm[key][j] = float(P33_lines[idx_wvl[0]].split()[j])
                    
                    P33[key] = P33_norm[key] * P11[key]
                    
                    # P43
                    P43_norm[key] = np.empty(len(P43_line0))
                    for j, theta in enumerate(P43_line0):
                        if i==0:
                            theta_P43_deg[j] = float(theta)
                        P43_norm[key][j] = float(P43_lines[idx_wvl[0]].split()[j])
                    
                    P43[key] = P43_norm[key] * P11[key]
                    
                    # P44
                    P44_norm[key] = np.empty(len(P44_line0))
                    for j, theta in enumerate(P44_line0):
                        if i==0:
                            theta_P44_deg[j] = float(theta)
                        P44_norm[key][j] = float(P44_lines[idx_wvl[0]].split()[j])
                    
                    P44[key] = P44_norm[key] * P11[key]

        if not self.HG:
            assert wvl0_exists
            # convert theta from degrees to radians
            self.theta_P11 = (theta_P11_deg * np.pi) / 180.
            self.theta_P12 = (theta_P12_deg * np.pi) / 180.
            self.theta_P22 = (theta_P22_deg * np.pi) / 180.
            self.theta_P33 = (theta_P33_deg * np.pi) / 180.
            self.theta_P43 = (theta_P43_deg * np.pi) / 180.
            self.theta_P44 = (theta_P44_deg * np.pi) / 180.
            
            self.P11 = P11
            self.P12 = P12
            self.P22 = P22
            self.P33 = P33
            self.P43 = P43
            self.P44 = P44
            
            P11_dat.close()
            P12_dat.close()
            P22_dat.close()
            P33_dat.close()
            P43_dat.close()
            P44_dat.close()        
        
        aspherical_optics.close()
        
        return(ssa_ice, ext_cff_mss_ice, g)
            
    def get_optical_properties(self, wvls, rds_snw):
        """ Retrieve snow optical properties from NetCDF files,
            based on user-specified wavelengths (wvls) and snow grain size
            (rds_snw)
            
            Returns ssa_ice, ext_cff_mss_ice, g, ssa_imp, ext_cff_mss_imp
        """
        self.snow_effective_radius = rds_snw
        # snow optics:
        fi_name = 'ice_wrn_%04d.nc' % rds_snw
        fi = os.path.join(self.optics_dir, 'mie', 'snicar', fi_name)
        snow_optics = netcdf.netcdf_file(fi, 'r')
        
        wvl_in = snow_optics.variables['wvl']
        ssa_in = snow_optics.variables['ss_alb']
        ext_in = snow_optics.variables['ext_cff_mss']
        asm_in = snow_optics.variables['asm_prm']
        
        if np.size(wvls)==1:
            wvl = wvls*1e-6
            
            # get indicies with smallest abs(wvl - wvl_in)
            idx_wvl = np.argsort(np.absolute(wvl - wvl_in.data))
            nearest_wvls = wvl_in[idx_wvl[:2]]
            
            # ssa_ice
            nearest_ssa_ice = ssa_in[idx_wvl[:2]]                
            try:
                if nearest_wvls[0] < nearest_wvls[1]:
                    ssa_ice_interp = interpolate.interp1d(nearest_wvls,
                                                          nearest_ssa_ice)
                    ssa_ice = ssa_ice_interp(wvl)
                else:
                    ssa_ice_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                          nearest_ssa_ice[::-1])
                    ssa_ice = ssa_ice_interp(wvl)
            except ValueError:
                ssa_ice = nearest_ssa_ice[0]
                sys.stderr.write('error: exception raised while interpolating '
                                 'ssa_ice, using nearest value instead\n')
            
            ssa_ice = np.array([ssa_ice])
            
            # ext_cff_mss_ice
            nearest_ext_cff_mss_ice = ext_in[idx_wvl[:2]]
            try:
                if nearest_wvls[0] < nearest_wvls[1]:
                    ext_cff_mss_ice_interp = interpolate.interp1d(nearest_wvls,
                                                            nearest_ext_cff_mss_ice)
                    ext_cff_mss_ice = ext_cff_mss_ice_interp(wvl)
                else:
                    ext_cff_mss_ice_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                      nearest_ext_cff_mss_ice[::-1])
                    ext_cff_mss_ice = ext_cff_mss_ice_interp(wvl)
            except ValueError:
                ext_cff_mss_ice = nearest_ext_cff_mss_ice[0]
                sys.stderr.write('error: exception raised while interpolating '
                                 'ext_cff_mss_ice, using nearest value instead\n')
            
            ext_cff_mss_ice = np.array([ext_cff_mss_ice])
            
            # g
            nearest_g = asm_in[idx_wvl[:2]]
            try:
                if nearest_wvls[0] < nearest_wvls[1]:
                    g_interp = interpolate.interp1d(nearest_wvls, nearest_g)
                    g = g_interp(wvl)
                else:
                    g_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                    nearest_g[::-1])
                    g = g_interp(wvl)
            except ValueError:
                g = nearest_g[0]
                sys.stderr.write('error: exception raised while interpolating '
                                 'g, using nearest value instead\n')
                                 
            g = np.array([g])
        
        elif np.size(wvls)>1:
            ssa_ice = np.empty(wvls.shape)
            ext_cff_mss_ice = np.empty(wvls.shape)
            g = np.empty(wvls.shape)     
            
            sorted_wvls = np.sort(wvls)
            last_wvl = None
            for i, wvl in enumerate(sorted_wvls):
                if wvl != last_wvl:
                    last_wvl = wvl
                    working_set_idxs = np.where(wvls==wvl)
                    
                    if wvl==self.wvl0:
                        self.wvl0_idx = working_set_idxs[0][0]                    
                    
                    wvl = wvl*1e-6
                    # get indicies with smallest abs(wvl - wvl_in)
                    idx_wvl = np.argsort(np.absolute(wvl - wvl_in.data))
                    nearest_wvls = wvl_in[idx_wvl[:2]]
            
                    # ssa_ice
                    nearest_ssa_ice = ssa_in[idx_wvl[:2]]
                    try:
                        if nearest_wvls[0] < nearest_wvls[1]:
                            ssa_ice_interp = interpolate.interp1d(nearest_wvls,
                                                                  nearest_ssa_ice)
                            ssa_ice[working_set_idxs] = ssa_ice_interp(wvl)
                        else:
                            ssa_ice_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                                  nearest_ssa_ice[::-1])
                            ssa_ice[working_set_idxs] = ssa_ice_interp(wvl)
                    except ValueError:
                        ssa_ice[working_set_idxs] = nearest_ssa_ice[0]
                        sys.stderr.write('error: exception raised while interpolating '
                                         'ssa_ice, using nearest value instead\n')
                    
                    # ext_cff_mss_ice
                    nearest_ext_cff_mss_ice = ext_in[idx_wvl[:2]]
                    try:
                        if nearest_wvls[0] < nearest_wvls[1]:
                            ext_cff_mss_ice_interp = interpolate.interp1d(nearest_wvls,
                                                                nearest_ext_cff_mss_ice)
                            ext_cff_mss_ice[working_set_idxs] = ext_cff_mss_ice_interp(wvl)
                        else:
                            ext_cff_mss_ice_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                          nearest_ext_cff_mss_ice[::-1])
                            ext_cff_mss_ice[working_set_idxs] = ext_cff_mss_ice_interp(wvl)
                    except ValueError:
                        ext_cff_mss_ice[working_set_idxs] = nearest_ext_cff_mss_ice[0]
                        sys.stderr.write('error: exception raised while interpolating '
                                         'ext_cff_mss_ice, using nearest value instead\n')
            
                    # g
                    nearest_g = asm_in[idx_wvl[:2]]
                    try:
                        if nearest_wvls[0] < nearest_wvls[1]:
                            g_interp = interpolate.interp1d(nearest_wvls, nearest_g)
                            g[working_set_idxs] = g_interp(wvl)
                        else:
                            g_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                            nearest_g[::-1])
                            g[working_set_idxs] = g_interp(wvl)
                    except ValueError:
                        g[working_set_idxs] = nearest_g[0]
                        sys.stderr.write('error: exception raised while interpolating '
                                         'g, using nearest value instead\n')
            
        #snow_optics.close()
        
        return(ssa_ice, ext_cff_mss_ice, g)
        
    #@timefunc
    def get_impurity_optics(self, wvls):
        """ fetch ssa and ext_cff_mss for impurities from self.fi_imp
        
            returns ssa_imp and ext_cff_mss_imp
        """
        fi_imp = os.path.join(self.optics_dir, 'mie', 'snicar', self.fi_imp)
        impurity_optics = netcdf.netcdf_file(fi_imp, 'r')
    
        wvl_in_imp = impurity_optics.variables['wvl']
        ssa_in_imp = impurity_optics.variables['ss_alb']
        ext_in_imp = impurity_optics.variables['ext_cff_mss']
    
        if np.size(wvls)==1:
            wvl = wvls*1e-6
        
            # get indicies with smallest abs(wvl - wvl_in_imp)
            idx_wvl = np.argsort(np.absolute(wvl - wvl_in_imp.data))
            nearest_wvls = wvl_in_imp[idx_wvl[:2]]
        
            # ssa_imp
            nearest_ssa_imp = ssa_in_imp[idx_wvl[:2]]
            try:
                if nearest_wvls[0] < nearest_wvls[1]:
                    ssa_imp_interp = interpolate.interp1d(nearest_wvls,
                                                          nearest_ssa_imp)
                    ssa_imp = ssa_imp_interp(wvl)
                else:
                    ssa_imp_interp = interpolate.interp1d(nearest_wvls[::-1], 
                                                          nearest_ssa_imp[::-1])
                    ssa_imp = ssa_imp_interp(wvl)
            except ValueError:
                ssa_imp = nearest_ssa_imp[0]
                sys.stderr.write('error: exception raised while '
                                 'interpolating ssa_imp, using nearest '
                                 'value instead\n')
        
            ssa_imp = np.array([ssa_imp])
        
            # ext_cff_mss_imp
            nearest_ext_cff_mss_imp = ext_in_imp[idx_wvl[:2]]
            try:
                if nearest_wvls[0] < nearest_wvls[1]:
                    ext_cff_mss_imp_interp = interpolate.interp1d(nearest_wvls,
                                                        nearest_ext_cff_mss_imp)
                    ext_cff_mss_imp = ext_cff_mss_imp_interp(wvl)
                else:
                    ext_cff_mss_imp_interp = interpolate.interp1d(nearest_wvls[::-1], nearest_ext_cff_mss_imp[::-1])
                    ext_cff_mss_imp = ext_cff_mss_imp_interp(wvl)
            except ValueError:
                ext_cff_mss_imp = nearest_ext_cff_mss_imp[0]
                sys.stderr.write('error: exception raised while '
                                 'interpolating ext_cff_mss_imp, using '
                                 'nearest value instead\n')
        
            ext_cff_mss_imp = np.array([ext_cff_mss_imp])
    
        elif np.size(wvls)>1:
            ssa_imp = np.empty(wvls.shape)
            ext_cff_mss_imp = np.empty(wvls.shape)
            
            sorted_wvls = np.sort(wvls)
            last_wvl = None
            for i, wvl in enumerate(sorted_wvls):
                if wvl != last_wvl:
                    last_wvl = wvl
                    working_set_idxs = np.where(wvls==wvl)
                    wvl = wvl*1e-6
        
                    # get indicies with smallest abs(wvl - wvl_in_imp)
                    idx_wvl = np.argsort(np.absolute(wvl - wvl_in_imp.data))
                    nearest_wvls = wvl_in_imp[idx_wvl[:2]]
        
                    # ssa_imp
                    nearest_ssa_imp = ssa_in_imp[idx_wvl[:2]]
                    try:
                        if nearest_wvls[0] < nearest_wvls[1]:
                            ssa_imp_interp = interpolate.interp1d(nearest_wvls,
                                                                  nearest_ssa_imp)
                            ssa_imp[working_set_idxs] = ssa_imp_interp(wvl)
                        else:
                            ssa_imp_interp = interpolate.interp1d(nearest_wvls[::-1], nearest_ssa_imp[::-1])
                            ssa_imp[working_set_idxs] = ssa_imp_interp(wvl)
                    except ValueError:
                        ssa_imp[working_set_idxs] = nearest_ssa_imp[0]
                        sys.stderr.write('error: exception raised while '
                                         'interpolating ssa_imp, using nearest '
                                         'value instead\n')
        
                    # ext_cff_mss_imp
                    nearest_ext_cff_mss_imp = ext_in_imp[idx_wvl[:2]]
                    try:
                        if nearest_wvls[0] < nearest_wvls[1]:
                            ext_cff_mss_imp_interp = interpolate.interp1d(nearest_wvls, nearest_ext_cff_mss_imp)
                            ext_cff_mss_imp[working_set_idxs] = ext_cff_mss_imp_interp(wvl)
                        else:
                            ext_cff_mss_imp_interp = interpolate.interp1d(nearest_wvls[::-1], nearest_ext_cff_mss_imp[::-1])
                            ext_cff_mss_imp[working_set_idxs] = ext_cff_mss_imp_interp(wvl)
                    except ValueError:
                        ext_cff_mss_imp[working_set_idxs] = nearest_ext_cff_mss_imp[0]
                        sys.stderr.write('error: exception raised while '
                                         'interpolating ext_cff_mss_imp, using '
                                         'nearest value instead\n')
                
        #impurity_optics.close()
    
        return(ssa_imp, ext_cff_mss_imp)
    
    def Henyey_Greenstein(self, g, costheta_p):
        """ Henyey-Greenstein scattering phase function
        """
        g = np.matrix(g).T # turn g into column vector
        g_2 = np.multiply(g,g) # compute g^2
        HG_num = 1 - g_2
        HG_den = np.power(1 + g_2 - 2*g*costheta_p, 3./2.)
        p_HG = np.asarray(HG_num / HG_den)
        
        return p_HG
    
    def Henyey_Greenstein2(self, g, costheta_p):
        """ Henyey Greenstein scattering phase function given asymmetry
            parameter g
        """
        if g==0:
            p_HG = 1 - 2*costheta_p
        else:
            p_HG = (1./(2.*g)) * (1 + g**2 - 
                                       ((1 - g**2)/(1 - g + 2*g*costheta_p))**2)
        
        return p_HG 
    
    def full_scattering_phase_function(self, P11, P12, stokes_params, theta, phi):
        """ Calculate the scattering phase function for arbitrarly polarized light
        """
        # unpack stokes parameters
        I = stokes_params[0]
        Q = stokes_params[1]
        U = stokes_params[2]
        V = stokes_params[3]
        
        P = np.empty((len(theta), len(phi)))
        for j, val in enumerate(phi):
            P[:,j] = P11*I + P12*(Q*np.cos(2*val) + U*np.sin(2*val))
            
        return P
    
    def interpolate_phase_matrix(self, wvls):
        """
        """
        self.P11_interp = dict()
        self.P12_interp = dict()
        self.P22_interp = dict()
        self.P33_interp = dict()        
        self.P43_interp = dict()
        self.P44_interp = dict()
        
        sorted_wvls = np.sort(wvls)
        last_wvl = None
        for wvl in sorted_wvls:
            if wvl != last_wvl:
                last_wvl = wvl
                self.P11_interp[wvl] = interpolate.interp1d(self.theta_P11, self.P11[wvl])
                self.P12_interp[wvl] = interpolate.interp1d(self.theta_P12, self.P12[wvl])
                self.P22_interp[wvl] = interpolate.interp1d(self.theta_P22, self.P22[wvl])
                self.P33_interp[wvl] = interpolate.interp1d(self.theta_P33, self.P33[wvl])
                self.P43_interp[wvl] = interpolate.interp1d(self.theta_P43, self.P43[wvl])
                self.P44_interp[wvl] = interpolate.interp1d(self.theta_P44, self.P44[wvl])
                
        # setup cutoff parameters for rejection method
        self.theta_cutoff = np.deg2rad(1)
        self.theta_max = self.theta_P11[-1]
        self.cutoff_idx = np.where(self.theta_P11==self.theta_cutoff)[0]
    
    def populate_pdfs(self, g, wvl, RANDOM_NUMBERS=1):
        """ 1. Populate PDF of cos(scattering phase angle) with random numbers
            2. Populate PDF of optical path traversed between scattering events
            3. Populate PDF of scattering azimuth angle with random numbers
            4. Populate PDF of single-scatter albedo with random numbers
            5. Populate PDF to determine extinction from ice or impurity
        
            Returns p_rand, tau_rand, phi_rand, ssa_rand, ext_spc_rand
        """
        p_rand = np.empty((g.size, RANDOM_NUMBERS))
        tau_rand = np.empty((g.size, RANDOM_NUMBERS))
        phi_rand = np.empty((g.size, RANDOM_NUMBERS))
        ssa_rand = np.empty((g.size, RANDOM_NUMBERS))
        ext_spc_rand = np.empty((g.size, RANDOM_NUMBERS))
        
        if self.shape != 'sphere' and not self.HG:
            I = self.stokes_params[0]
            Q = self.stokes_params[1]
            U = self.stokes_params[2]
            V = self.stokes_params[3]
            
            if Q == 0:
                beta = 0.5 * np.arctan(U / 1e-15)
            else:
                beta = 0.5 * np.arctan(U / Q)
        
        for i, val in enumerate(wvl):
            if self.shape == 'sphere' or self.HG:
                # 1. Populate PDF of cos(scattering phase angle) with random numbers
                r1 = np.random.rand(RANDOM_NUMBERS) # distribution from 0 -> 1
                p_rand[i, :] = self.Henyey_Greenstein2(g[i], r1)
                
                # 3. Populate PDF of scattering azimuth angle with random numbers
                phi_rand[i,:] = np.random.rand(RANDOM_NUMBERS) * TWO_PIE # 0 -> 2pi
            else:
                
                P11_interp = self.P11_interp[val]
                P12_interp = self.P12_interp[val]
                
                # find max val in area 1
                A = np.absolute(I) * np.absolute(self.P11[val][:self.cutoff_idx]).max()
                B = np.absolute(self.P12[val][:self.cutoff_idx]).max()
                C = np.absolute(Q * np.cos(2*beta) + U * np.sin(2*beta))
                
                max_val1 = A + B * C
                
                # find max val in area 2
                A = np.absolute(I) * np.absolute(self.P11[val][self.cutoff_idx:]).max()
                B = np.absolute(self.P12[val][self.cutoff_idx:]).max()
                #C = np.absolute(Q * np.cos(2*beta) + U * np.sin(2*beta)) # C is the same in both areas
                
                max_val2 = A + B * C

                area1 = max_val1 * self.theta_cutoff
		area2 = max_val2 * (self.theta_max - self.theta_cutoff)
		percent_area1 = area1 / (area1 + area2)
                
                area_rand = np.random.rand(RANDOM_NUMBERS)
	                                                          
                theta_rand = np.random.rand(RANDOM_NUMBERS) # 0 -> 1
                phi_rand[i, :] = np.random.rand(RANDOM_NUMBERS) * TWO_PIE # 0 -> 2pi
                two_phi_rand = 2 * phi_rand[i, :]
                for j, theta in enumerate(theta_rand):
                    #print(i,j)
	            two_phi = two_phi_rand[j]
		    area_rand_j = area_rand[j]
                    if area_rand_j <= percent_area1:
                        area = 1
                        theta = theta * self.theta_cutoff # 0 -> theta_cutoff
                    else:
                        area = 2
                        theta = theta * (self.theta_max - self.theta_cutoff) + self.theta_cutoff

                    # rejection method
                    r3 = 1
                    phase_func_val = 0
                    k = 0
                    while r3 > phase_func_val:
                        """ WHILE LOOP MUST BE EFFICIENT
                        """
                        if k > 0:
                            area_rand_j = np.random.rand()
                            if area_rand_j <= percent_area1:
                                area = 1
                                theta = np.random.rand() * self.theta_cutoff # 0 -> theta_cutoff
                            else:
                                area = 2
                                theta = (np.random.rand() * (self.theta_max - self.theta_cutoff) + 
                                                            self.theta_cutoff)
                            two_phi = np.random.rand() * FOUR_PIE
                        if area == 1:
                            r3 = np.random.rand() * max_val1
                        if area == 2:
                            r3 = np.random.rand() * max_val2
                    
                        S11 = P11_interp(theta)
                        S12 = P12_interp(theta)
                        phase_func_val = I*S11 + S12 * (Q*np.cos(two_phi) +
                                                        U*np.sin(two_phi))
                        
                        k += 1
                    theta_rand[j] = theta
                    phi_rand[i,j] = two_phi / 2.
                p_rand[i,:] = np.cos(theta_rand)

            # SANITY CHECK:  mean of the random distribution (should equal g)
            #p_mean = np.mean(p_rand[i,:])
            #print p_mean - g[i]
            
            # 2. Populate PDF of optical path traversed between scattering 
            #    events
            # Q1:  After an optical path of tau, how many photons have NOT 
            #      suffered an extinction event?
            tau_rand[i,:] = -np.log(np.random.rand(RANDOM_NUMBERS))
        
            # median of tau_rand should be -log(0.5)=0.6931
            #tau_median = np.median(tau_rand)
        
            # 4. Populate PDF of single-scatter albedo with random numbers
            ssa_rand[i,:] = np.random.rand(RANDOM_NUMBERS) # 0 -> 1
        
            # 5. Populate PDF to determine extinction from ice or impurity
            ext_spc_rand[i,:] = np.random.rand(RANDOM_NUMBERS) # 0 -> 1                          
        
        return(p_rand, tau_rand, phi_rand, ssa_rand, ext_spc_rand)
    
    def initial_pdfs(self, wvls, RANDOM_NUMBERS=1):
        """ populate pdfs only needed for photon's first loop
        """
        p_rand = np.empty((wvls.size, RANDOM_NUMBERS))
	phi_rand = np.empty((wvls.size, RANDOM_NUMBERS))
	tau_rand = np.empty((wvls.size, RANDOM_NUMBERS))
        ssa_rand = np.empty((wvls.size, RANDOM_NUMBERS))
        ext_spc_rand = np.empty((wvls.size, RANDOM_NUMBERS))
        for i, val in enumerate(wvls):
            tau_rand[i,:] = -np.log(np.random.rand(RANDOM_NUMBERS))
            ssa_rand[i,:] = np.random.rand(RANDOM_NUMBERS) # 0 -> 1
            ext_spc_rand[i,:] = np.random.rand(RANDOM_NUMBERS) # 0 -> 1
        
        self.p_rand = p_rand
	self.phi_rand = phi_rand
	self.tau_rand = tau_rand
        self.ssa_rand = ssa_rand
        self.ext_spc_rand = ext_spc_rand
            
    def scatter_photon(self, i, dtau_current, theta_sca, phi_sca):
        """
        """
        x_tau = self.x_tau
        y_tau = self.y_tau
        z_tau = self.z_tau
    
        mux_0 = self.mux_0
        muy_0 = self.muy_0
        muz_0 = self.muz_0
    
        sintheta = np.sin(theta_sca * (np.pi / 180.))
        costheta = np.cos(theta_sca * (np.pi / 180.))
        sinphi = np.sin(phi_sca * (np.pi / 180.))
        cosphi = np.cos(phi_sca * (np.pi / 180.))

        if muz_0==1:
            mux_n = sintheta * cosphi
            muy_n = sintheta * sinphi
            muz_n = costheta
        elif muz_0==-1:
            mux_n = sintheta * cosphi
            muy_n = -sintheta * sinphi
            muz_n = -costheta
        else:
            mux_n = ((sintheta*(mux_0*muz_0*cosphi - 
                                muy_0*sinphi)) / (np.sqrt(1 - muz_0**2)) + 
                     mux_0*costheta)
            muy_n = ((sintheta*(muy_0*muz_0*cosphi + 
                                mux_0*sinphi)) / (np.sqrt(1 - muz_0**2)) + 
                     muy_0*costheta)
            muz_n = -np.sqrt(1 - muz_0**2)*sintheta*cosphi + muz_0*costheta

        # update coordinates:
        self.x_tau = np.append(x_tau, x_tau[i-1] + dtau_current*mux_n)
        self.y_tau = np.append(y_tau, y_tau[i-1] + dtau_current*muy_n)
        self.z_tau = np.append(z_tau, z_tau[i-1] + dtau_current*muz_n)

        # update current direction (0):
        self.mux_0 = mux_n
        self.muy_0 = muy_n
        self.muz_0 = muz_n
    
    def rotate_stokes_vector(self, angle, stokes_params):
        """ rotate stokes vector into new frame of referecne by angle (angle)
        """
        I_old = stokes_params[0]
        Q_old = stokes_params[1]
        U_old = stokes_params[2]
        V_old = stokes_params[3]
        
        double_angle = 2*angle
        cos_double_angle = np.cos(double_angle)
        sin_double_angle = np.sin(double_angle)
        
        I_new = I_old
        Q_new = Q_old * cos_double_angle + U_old * sin_double_angle
        U_new = -Q_old * sin_double_angle + U_old * cos_double_angle
        V_new = V_old
        
        rotated_stokes_params = (I_new, Q_new, U_new, V_new)
        
        return rotated_stokes_params
    
    def monte_carlo3D(self, wvl):
        """ Translated from matlab to python by Adam Schneider
        """        
        i_max = 1
        # initialization:
        x_tau = np.array([0])
        y_tau = np.array([0])
        z_tau = np.array([0])
        
        # initial direction cosines (phi_0 = 0)
        mux_0 = np.sin(self.theta_0)
        muy_0 = 0
        muz_0 = -np.cos(self.theta_0)
            
        x_crt = np.array([0])
        y_crt = np.array([0])
        z_crt = np.array([0])
        
        path_length = 0
        
        if self.shape != 'sphere' and not self.HG:
            P11_interp = self.P11_interp[wvl]
            P12_interp = self.P12_interp[wvl]
            P22_interp = self.P22_interp[wvl]
            P33_interp = self.P33_interp[wvl]
            P43_interp = self.P43_interp[wvl]
            P44_interp = self.P44_interp[wvl]
        
        if self.debug: # debugging / demonstration of 2 scattering events:
            self.x_tau = x_tau
            self.y_tau = y_tau
            self.z_tau = z_tau
            
            self.mux_0 = mux_0
            self.muy_0 = muy_0
            self.muz_0 = muz_0            
            
            # 1. photon enters from above
            i = 1
            dtau_current = 0.2
            theta_sca = 0
            phi_sca = 50
            self.scatter_photon(i, dtau_current, theta_sca, phi_sca)
            
            # 2. photon is scattered in some random direction
            i = 2
            dtau_current = 0.4
            theta_sca = 20
            phi_sca = 50
            self.scatter_photon(i, dtau_current, theta_sca, phi_sca)
            
            x_tau = self.x_tau
            y_tau = self.y_tau
            z_tau = self.z_tau
            
            if self.flg_3D==0: # 2-D plot:
                # 2-D plot:
                fig = plt.figure()
                plt.plot(x_tau, z_tau, linewidth=3)
                plt.xlim(-0.14, 0.14)
                plt.ylim(-0.5, 0)
                plt.xlabel('Optical Depth (x)', fontsize=18)
                plt.ylabel('Optical Depth (z)', fontsize=18)
                plt.grid()
                plt.show()
            
            elif self.flg_3D==1: # 3-D plot
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot(x_tau, y_tau, z_tau, linewidth=3)
                ax.set_xlabel('Optical Depth (x)', fontsize=18)
                ax.set_ylabel('Optical Depth (y)', fontsize=18)
                ax.set_zlabel('Optical Depth (z)', fontsize=18)
                plt.show()
         
            # reinitialize
            # initialization:
            x_tau = np.array([0])
            y_tau = np.array([0])
            z_tau = np.array([0])
        
            # initial direction cosines
            mux_0 = np.sin(self.theta_0)
            muy_0 = 0
            muz_0 = -np.cos(self.theta_0)
            
            x_crt = np.array([0])
            y_crt = np.array([0])
            z_crt = np.array([0])
            
            path_length = 0
        
        # scatter the photon inside the cloud/snow until it escapes or is 
        # absorbed
        bottom_reflection = False
        condition = 0
        i = 0
        i_rand = 0
        while condition==0:
            i+=1
            i_rand+=1
            if i_rand > i_max: # we need more random numbers!
                g = np.array([self.g[self.photon]])
                pdfs = self.populate_pdfs(g, np.array([wvl]))
                self.p_rand[self.photon] = pdfs[0][0]
                self.tau_rand[self.photon] = pdfs[1][0]
                self.phi_rand[self.photon] = pdfs[2][0]
                self.ssa_rand[self.photon] = pdfs[3][0]
                self.ext_spc_rand[self.photon] = pdfs[4][0]
                
                i_rand = 1
            
            # distance, in optical depth space, to move photon
            dtau_current = self.tau_rand[self.photon, i_rand-1]
            if self.Lambertian_surface and i==1: # photon scattered at surface
                dtau_current = 0
            
            # scattering phase angle:
            if i==1: # the photon enters snow at zenith angle theta_0 with no 
                      # deflection 
                costheta = 1
                sintheta = 0
                cosphi = 0
                sinphi = 0
            elif self.Lambertian_surface or bottom_reflection:
                valid_val=0
                while valid_val==0:
                    theta_rand = np.random.uniform(0, np.pi/2)
                    r1 = np.random.rand()
                    if r1 < 2 * np.sin(theta_rand) * np.cos(theta_rand):
                        costheta = np.cos(theta_rand + np.arccos(muz_0))
                        sintheta = np.sqrt(1 - costheta**2)
                        valid_val=1
                bottom_reflection = False
            else:
                costheta = self.p_rand[self.photon, i_rand-1]
                sintheta = np.sqrt(1 - costheta**2)
                
            if i > 1:
                # scattering azimuth angle:
                cosphi = np.cos(self.phi_rand[self.photon, i_rand-1])
                sinphi = np.sin(self.phi_rand[self.photon, i_rand-1])
            
                # new cosine directional angles                
                if muz_0==1:
                    mux_n = sintheta * cosphi
                    muy_n = sintheta * sinphi
                    muz_n = costheta
                elif muz_0==-1:
                    mux_n = sintheta * cosphi
                    muy_n = -sintheta * sinphi
                    muz_n = -costheta
                else: # equations from  http://en.wikipedia.org/wiki/Monte_Carlo_method_for_photon_transport
                    mux_n = ((sintheta*(mux_0*muz_0*cosphi - muy_0*sinphi)) /
                             (np.sqrt(1 - muz_0**2)) + mux_0*costheta)
                    muy_n = ((sintheta*(muy_0*muz_0*cosphi + mux_0*sinphi)) /
                             (np.sqrt(1 - muz_0**2)) + muy_0*costheta)
                    muz_n = -np.sqrt(1 - muz_0**2)*sintheta*cosphi + muz_0*costheta
            
                if self.shape != 'sphere' and not self.HG:
                    # update stokes paramters via scattering phase matrix
                    theta_sca = np.arccos(costheta)
                    phi_sca = np.arccos(cosphi)
                    
                    # step 1 - Rotation of the reference frame into the scattering plane
                    (I_sp,
                     Q_sp,
                     U_sp,
                     V_sp) = self.rotate_stokes_vector(phi_sca, self.stokes_params)
                    
                    # step 2 - Scattering of the photon at an angle theta_sca in the
                    #          scattering plane
                    P11 = P11_interp(theta_sca)
                    P12 = P12_interp(theta_sca)
                    P22 = P22_interp(theta_sca)
                    P33 = P33_interp(theta_sca)
                    P43 = P43_interp(theta_sca)
                    P44 = P44_interp(theta_sca)
                    
                    I_sca = I_sp * P11 + Q_sp * P12
                    Q_sca = I_sp * P12 + Q_sp * P22
                    U_sca = U_sp * P33 - V_sp * P43
                    V_sca = U_sp * P43 + V_sp * P44
                    
                    # step 3 - Return the reference frame to a new meridian plane
                    num = muz_n * np.cos(theta_sca) - muz_0
                    if phi_sca >= np.pi:
                        den = np.sqrt((1 - costheta**2)*(1 - muz_n**2))
                    elif phi_sca < np.pi:
                        den = -np.sqrt((1 - costheta**2)*(1 - muz_n**2))
                        
                    gama = np.arccos(num / den)
                    
                    stokes_sca = (I_sca, Q_sca, U_sca, V_sca)
                    
                    (I_merd,
                     Q_merd, 
                     U_merd,
                     V_merd) = self.rotate_stokes_vector(-gama, stokes_sca)
                     
                    self.stokes_params = (np.array([I_merd, Q_merd, U_merd, V_merd]) /
                                          I_merd)
                    
            elif i==1:
                mux_n = mux_0
                muy_n = muy_0
                muz_n = muz_0      
            
            # update tau coordinates:
            x_tau = np.append(x_tau, x_tau[i-1] + dtau_current*mux_n)
            y_tau = np.append(y_tau, y_tau[i-1] + dtau_current*muy_n)
            z_tau = np.append(z_tau, z_tau[i-1] + dtau_current*muz_n)
                        
            # update Cartesian coordinates:
            ext_cff_mss = self.ext_cff_mss[self.photon]
            ext_cff = ext_cff_mss * self.rho_snw
            
            multiplier = dtau_current / ext_cff
            
            x_crt = np.append(x_crt, x_crt[i-1] + 
                                     multiplier * mux_n)                            
            y_crt = np.append(y_crt, y_crt[i-1] + 
                                     multiplier * muy_n)
            z_crt = np.append(z_crt, z_crt[i-1] + 
                                     multiplier * muz_n)
                              
            if i > 1:
                # update current direction:
                mux_0 = mux_n
                muy_0 = muy_n
                muz_0 = muz_n
            
            # update path length
            path_length += dtau_current / ext_cff
                                                        
            # was the extinction event caused by ice or impurity?
            if self.ext_spc_rand[self.photon, i_rand-1] > self.P_ext_imp[self.photon]:
                # extinction from ice
                ext_state = 1
                ssa_event = self.ssa_ice[self.photon]
            else:
                # extinction from impurity
                ext_state = 2
                ssa_event = self.ssa_imp[self.photon]
            
            if self.Lambertian: # set ssa_event to Lambertian reflectance
                 ssa_event = self.R_Lambertian
                        
            # check for exit status:
            if z_tau[i] > 0:
                # photon has left the top of the cloud/snow (reflected)
                condition = 1
                # correct path_length (we only want photon path length within
                # the snow pack)
                correction = -((z_tau[i] * dtau_current) /
                               ((z_tau[i] - z_tau[i-1]) * ext_cff))
                path_length += correction
                
            elif z_tau[i] < -self.tau_tot and i==1:
                # photon has left the bottom of the cloud/snow WITHOUT 
                # scattering ONCE (direct transmittance)
                # correct path_length (we only want photon path length within
                # the snow pack)
                correction = -(((z_tau[i] + self.tau_tot) * dtau_current) /
                               ((z_tau[i] - z_tau[i-1]) * ext_cff))
                path_length += correction                
                
                # correct coordinates for reflection off bottom
                dtau_correction = -(((z_tau[i] + self.tau_tot)/
                                     (z_tau[i-1] - z_tau[i])) * dtau_current)
                                     
                x_tau[i] = x_tau[i] - (mux_n * dtau_correction)
                y_tau[i] = y_tau[i] - (muy_n * dtau_correction)
                z_tau[i] = z_tau[i] - (muz_n * dtau_correction)
                
                multiplier = dtau_correction / ext_cff
                x_crt[i] = x_crt[i] - (mux_n * multiplier)
                y_crt[i] = y_crt[i] - (muy_n * multiplier)
                z_crt[i] = z_crt[i] - (muz_n * multiplier)
                
                if self.Lambertian_bottom:
                    reflectance_rand = np.random.rand() # 0 -> 1
                    if reflectance_rand <= self.R_Lambertian:
                        bottom_reflection = True
                    else:
                        condition = 3
                else:
                    condition = 3
                
            elif z_tau[i] < -self.tau_tot:
                # photon has left the bottom of the cloud/snow (diffuse 
                # transmittance)
                # correct path_length (we only want photon path length within
                # the snow pack)
                correction = -(((z_tau[i] + self.tau_tot) * dtau_current) /
                               ((z_tau[i] - z_tau[i-1]) * ext_cff))
                path_length += correction
                
                # correct coordinates for reflection off bottom
                dtau_correction = -(((z_tau[i] + self.tau_tot)/
                                     (z_tau[i-1] - z_tau[i])) * dtau_current)
                                     
                x_tau[i] = x_tau[i] - (mux_n * dtau_correction)
                y_tau[i] = y_tau[i] - (muy_n * dtau_correction)
                z_tau[i] = z_tau[i] - (muz_n * dtau_correction)
                
                multiplier = dtau_correction / ext_cff
                x_crt[i] = x_crt[i] - (mux_n * multiplier)
                y_crt[i] = y_crt[i] - (muy_n * multiplier)
                z_crt[i] = z_crt[i] - (muz_n * multiplier)
                
                if self.Lambertian_bottom:
                    reflectance_rand = np.random.rand() # 0 -> 1
                    if reflectance_rand <= self.R_Lambertian:
                        bottom_reflection = True
                    else:
                        condition = 2
                else:
                    condition = 2
            
            elif self.ssa_rand[self.photon, i_rand-1] >= ssa_event:
                # photon was absorbed, archive which species absorbed it:
                if ext_state==1:
                    condition = 4
                elif ext_state==2:
                    condition = 5

        wvn = 1. / wvl
        theta_n = np.arccos(muz_0)
        if i==1:
            phi_n = 0.
        else:
            phi_n = np.arctan(muy_0 / mux_0)
        n_scat = i-1 # number of scattering events
        
        return(condition, wvn, theta_n, phi_n, n_scat, path_length)
              
    def run(self, n_photon, wvl0, half_width, rds_snw, theta_0=0.,
            stokes_params=np.array([1,0,0,0]), shape='sphere',
            roughness='smooth', test=False, debug=False, Lambertian_surface=False,
            Lambertian_bottom=True, Lambertian_reflectance=1.):
        """ Run the Monte Carlo model given a normal distribution of
            wavelengths [um].  This better simulates what NERD does with
            non-monochromatic LEDs.
            
            ALL VALUES IN MICRONS
        """
        self.debug = debug
        self.Lambertian_surface = Lambertian_surface
        self.Lambertian_bottom = Lambertian_bottom
        self.R_Lambertian = Lambertian_reflectance
        
        self.theta_0 = (np.pi * theta_0) / 180. # theta_0 deg -> rad
        self.shape = shape
        self.roughness = roughness
        
        self.wvl0 = wvl0
        self.initial_stokes_params = stokes_params
        self.stokes_params = stokes_params
        
        # Convert half_width to standard deviation
        scale = half_width / 2.355
        
        # Generate random array of photon wavelengths, rounded to nearest 10 nm
        wvls = np.around(np.random.normal(loc=wvl0, scale=scale,
                                          size=(n_photon)), decimals=2)
        par_wvls = Parallel(wvls)
        
        # get ice optical data
        if shape=='sphere':
            (ssa_ice,
             ext_cff_mss_ice,
             g) = self.get_optical_properties(par_wvls.working_set, rds_snw)
             
        elif wvl0 >= 0.2 and wvl0 <= 15.25:
            self.far_IR = False
            (ssa_ice,
             ext_cff_mss_ice,
             g) = self.get_aspherical_SSPs(par_wvls.working_set, rds_snw)
             
            if not self.HG: # reinstantiate par_wvls for new wvls
                par_wvls.working_set = self.wvls
            
        elif wvl0 >= 16.4 and wvl0 <= 99.0:
            self.far_IR = True
            (ssa_ice,
             ext_cff_mss_ice,
             g) = self.get_aspherical_SSPs(par_wvls.working_set, rds_snw)
             
            if not self.HG: # reinstantiate par_wvls for new wvls
                par_wvls.working_set = self.wvls
             
        # get impurity optical data                       
        (ssa_imp,
         ext_cff_mss_imp) = self.get_impurity_optics(par_wvls.working_set)
        
        #print self.snow_effective_radius
        
        if test:
            try:
                ssa_ice = self.ssa_ice * np.ones(n_photon)
            except AttributeError:
                pass
            try:
                ext_cff_mss_ice = self.ext_cff_mss_ice * np.ones(n_photon)
            except AttributeError:
                pass
            try:
                g = self.g * np.ones(n_photon)
            except AttributeError:
                pass
            try:
                ssa_imp = self.ssa_imp * np.ones(n_photon)
            except AttributeError:
                pass
            try:
                ext_cff_mss_imp = self.ext_cff_mss_imp * np.ones(n_photon)
            except AttributeError:
                pass
                
        imp_cnc = self.imp_cnc
        # combined mass extinction cross-section of ice+impurity system (m2/kg)
        self.ext_cff_mss = ext_cff_mss_ice*(1-imp_cnc) + ext_cff_mss_imp*imp_cnc
        
        # calculate probability that extinction event is caused by impurity
        self.P_ext_imp = (imp_cnc*ext_cff_mss_imp) / (imp_cnc*ext_cff_mss_imp + 
                                                    (1-imp_cnc)*ext_cff_mss_ice)

        # cos(theta) array over which to compute function
        #self.costheta_p = np.arange(-1.000, 1.001, 0.001)
        self.g = g
        #self.p = self.Henyey_Greenstein()
        self.ssa_ice = ssa_ice
        self.ssa_imp = ssa_imp
        
        if not self.phase_functions:
            if shape != 'sphere' and not self.HG:
                self.interpolate_phase_matrix(par_wvls.working_set)
            
            self.initial_pdfs(par_wvls.working_set)
            """ Deprecated, only use for testing     
            (self.p_rand,
             self.tau_rand,
             self.phi_rand,
             self.ssa_rand,
             self.ext_spc_rand) = self.populate_pdfs(g, par_wvls.working_set)
            """ 
            # counters for saving coordinates of absorption events and exit_top 
            # events
            #self.i1 = 1
            #self.i2 = 1
            #self.i_sensor = 0
       
            answer = list() 
            for i, wvl in enumerate(par_wvls.working_set):
                self.photon = i
                answer.append(self.monte_carlo3D(wvl))
                
            all_answers = par_wvls.answer_and_reduce(answer,
                                                     MonteCarlo.flatten_list)
            if all_answers is not None:
                # this is the root processor
                output_file = self.setup_output(n_photon, wvl0, half_width, rds_snw)
                txt_file = open(output_file, 'w')
                txt_file.write('condition wvn[um^-1] theta_n phi_n n_scat '
                               'path_length[m]\n')
                for i, answer in enumerate(all_answers):
                    txt_file.write('%d %r %r %r %d %r\n' % (answer[0], answer[1],
                                                            answer[2], answer[3],
                                                            answer[4], answer[5]))
                txt_file.close()
                print('%s' % output_file) # for easy post processing
        
        else: # no modeling, only plotting phase functions
            self.plot_phase_function()
            if not self.shape=='sphere' and not self.HG:
                # also compare phase functions
                self.plot_phase_functions()
                                                      
    def calculate_albedo(self, answers):
        """ Compute black sky albedo (directional-hemispherical reflectance) of
            the snow/cloud
        """
        Q_down = 0
        Q_up = 0
        for photon, answer in enumerate(answers):
            Q_down += answer[1]
            if answer[0] == 1:
                Q_up += answer[1]
        albedo = Q_up / Q_down
                
        return albedo
    
    def plot_phase_functions(self):
        """ Plot azimuthally averaged scattering phase function along with
            the Henyey-Greenstein phase function for comparison 
        """
        g = self.g[self.wvl0_idx]
        costheta_p = np.arange(-1.000, 1.001, 0.001)        
        P_HG = self.Henyey_Greenstein(g, costheta_p)[0]
        
        phi = np.arange(0, 2*np.pi, np.pi / 1800.)
        P_full = self.full_scattering_phase_function(self.P11[self.wvl0],
                                                     self.P12[self.wvl0],
                                                     self.initial_stokes_params,
                                                     self.theta_P11, phi)
        costheta_full = np.cos(self.theta_P11)
        P_full_means = np.mean(P_full, axis=1)
        
        fig = plt.figure()
        g_rounded = np.around(g, 4)
        plt.semilogy(costheta_full, P_full_means, label='Full scattering phase function')
        plt.semilogy(costheta_p, P_HG, label='Henyey-Greenstein phase function')
        
        plt.xlabel(r'$\cos(\theta)$', fontsize=18)
        plt.ylabel('Relative probability', fontsize=18)
        plt.xlim((-1.01, 1.01))
        plt.grid()
        plt.legend(loc=2)
        plt.title('Scattering phase functions (g = %s)' % g_rounded, fontsize=18)
        
        plt.show()
        
        theta_p = np.arccos(costheta_p)
        theta_p_2 = -theta_p + 2*np.pi
        theta_p = np.array([theta_p, theta_p_2]).flatten()
        P_HG = np.array([P_HG, P_HG]).flatten()
        
        theta_P11_2 =  -self.theta_P11 + 2*np.pi 
        theta_P11 = np.array([self.theta_P11, theta_P11_2]).flatten()
        P_full_means = np.array([P_full_means, P_full_means]).flatten()
        
        log_P_HG = np.log10(P_HG)
        log_P_full_means = np.log10(P_full_means)
        
        plt.polar(theta_P11, log_P_full_means, label='Full scattering phase function')
        plt.polar(theta_p, log_P_HG, label='Henyey-Greenstein phase function')
        plt.legend()
        plt.title('Scattering phase functions (g = %s)' % g_rounded, fontsize=18)
        
        plt.show()
    
    #@timefunc          
    def plot_phase_function(self):
        """ plot phase function versus cos(theta)
        
            For computational purposes, will only plot the first 100 curves.
            The first 100 curves are random, so this will give a good sample 
            for larger N
        """
        if self.shape=='sphere' or self.HG:
            costheta_p = np.arange(-1.000, 1.001, 0.001)
            P = self.Henyey_Greenstein(self.g[:100], costheta_p)
            fig = plt.figure()
            if np.size(self.g)>1:
                mean_g = np.around(np.mean(self.g), 4)
                std_g = np.around(np.std(self.g), 4)
                for i, val in enumerate(self.g):
                    if i < 100:
                        plt.semilogy(costheta_p, P[i])
                    
                        plt.title('Henyey-Greenstein Phase Function for\n'
                                  'mean(g) = %s and std(g) = %s' % (mean_g, std_g),
                                  fontsize=18)
            elif np.size(self.g)==1:
                g_rounded = np.around(self.g, 4)
                plt.semilogy(costheta_p, P[0])
            
                plt.title('Henyey-Greenstein Phase Function for\n'
                          'g = %s' % g_rounded[0], fontsize=18)
            plt.xlabel(r'$\cos(\theta)$', fontsize=18)
            plt.ylabel('Relative probability', fontsize=18)
            plt.xlim((-1, 1))
            plt.grid()
        
            plt.show()
        
        else: # plot full scattering phase function for first wvl and initial
              # stokes params
            phi = np.arange(0, 2*np.pi, np.pi / 1800.)
            P = self.full_scattering_phase_function(self.P11[self.wvl0], self.P12[self.wvl0],
                                                    self.initial_stokes_params,
                                                    self.theta_P11, phi)
            
            theta_deg = (self.theta_P11 * 180.) / np.pi
            log_P = np.log10(P) # log scale P
            nlevels = 256
            fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
            cax = ax.contourf(phi, theta_deg, log_P, nlevels)
            
            plt.show()
    
    def get_model_args(self):    
        """ Specify model kwargs at run time or get values from config.ini
        """  
        config = ConfigParser.SafeConfigParser()
        #data_dir = os.path.join(os.getcwd(), 'data')
        #config_file = os.path.join(data_dir, 'config.ini')
        config_file = 'config.ini'
        config.read(config_file)
        
        # model parameters
        section_name = 'model parameters'
        tau_tot = config.getfloat(section_name, 'tau_tot')
        imp_cnc = config.getfloat(section_name, 'imp_cnc')
        rho_snw = config.getfloat(section_name, 'rho_snw')
        rho_ice = config.getfloat(section_name, 'rho_ice')
        #rsensor = config.getfloat(section_name, 'rsensor')
        #hsensor = config.getfloat(section_name, 'hsensor')
        
        # plot options
        section_name = 'plot options'
        flg_crt = config.getint(section_name, 'flg_crt')
        flg_3D = config.getint(section_name, 'flg_3D')

        # data
        section_name = 'data'
        output_dir = config.get(section_name, 'output_dir')
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
        parser.add_argument('--rho_ice', type=float, default=rho_ice,
                            help='ice density (kg/m3)')
        parser.add_argument('--rsensor', type=float, default=None,
                            help='sensor radius [m]') 
        parser.add_argument('--hsensor', type=float, default=None,
                            help='sensor height above snow [m]')
        parser.add_argument('--flg_crt', type=int, default=flg_crt,
                            help='plot in optical depth space (=0) or '
                                 'Cartesian space (=1)?')
        parser.add_argument('--flg_3D', type=int, default=flg_3D,
                            help='plot in 2-D (=0), 3-D (=1). '
                                 'or no plot (=999)?')
        parser.add_argument('--output_dir', type=str, default=output_dir,
                            help='directory to write output data')
        parser.add_argument('--optics_dir', type=str, default=optics_dir, 
                            help='directory of optics files')
        parser.add_argument('--fi_imp', type=str, default=fi_imp)
        parser.add_argument('--HG', action='store_true', help='Use '
                            'Henyey-Greenstein phase function instead of full '
                            'scattering phase matrix (this is done '
                            'automatically for spherical particles)')
        parser.add_argument('--phase_functions', action='store_true',
                            help='Plot phase functions')
        
        args = parser.parse_args()
        
        return args

    @classmethod
    def flatten_list(klass, l):            
        return [item for sublist in l for item in sublist]

def test(n_photon=50000, wvl=0.5, half_width=0.085, rds_snw=100):
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
    #test_case = MonteCarlo(tau_tot=tau_tot, imp_cnc=imp_cnc, phase_functions=True)
    test_case.ssa_ice = ssa_ice
    test_case.g = g
    
    test_case.run(n_photon, wvl, half_width, rds_snw, test=True)
    
def test_and_debug(n_photon=100, wvl=0.5, half_width=0.085, rds_snw=250):
    """ manually specify optical properties for test cases and debugging:
    """
    # mass extinction cross-section of ice grains (m2/kg at 500nm)
    ext_cff_mss_ice = 6.6 # typical for re=250um
    
    # single-scatter albedo of ice grains (at 500nm)
    ssa_ice = 0.999989859099 # typical for re=250um
    
    # scattering asymmetry parameter for ice grains
    g = -0.89 # typical for re=250
    
    # mass extinction cross-section of black carbon (m2/kg at 500nm)
    ext_cff_mss_imp = 12000
    
    # single-scatter albedo of black carbon (500nm)
    ssa_imp = 0.30
    
    test_case = MonteCarlo(tau_tot=10)
    #test_case = MonteCarlo(tau_tot=10, phase_functions=True)
    test_case.ext_cff_mss_ice = ext_cff_mss_ice
    test_case.ssa_ice = ssa_ice
    test_case.g = g
    test_case.ext_cff_mss_imp = ext_cff_mss_imp
    test_case.ssa_imp = ssa_imp
    
    test_case.run(n_photon, wvl, half_width, rds_snw, test=True)

def run():
    """ USER INPUT
    """
    # Run model or plot phase functions?
    phase_functions = False
    
    # set number of photons
    n_photon = 100
    
    # wavelength [um]
    wvl = 1.3
    #wvl = 1.55
    wvl = 0.3
    
    # half width [um]
    half_width = 0.085
    half_width = 1e-15
    
    # snow effective grain size [um]
    rds_snw = 100.
    
    # snow optical depth
    tau_tot = 1000.0
    
    # mass concentration of impurity [mIMP/(mIMP+mICE)]
    imp_cnc=0e-9
    
    # plot in optical depth space (=0) or Cartesian space (=1)?
    flg_crt = 1
    
    # plot in 2-D (=0), 3-D (=1). or no plot (=999)?
    flg_3D = 999

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
                                   fi_imp=fi_imp,
                                   phase_functions=phase_functions)
                                   
    monte_carlo_model.run(n_photon, wvl, half_width, rds_snw)

def main():
    run()

if __name__=='__main__':
    main()
