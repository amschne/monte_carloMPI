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
#from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

from parallelize import Parallel

#import ipdb
#from memory_profiler import profile

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
            output_dir = [DIRECTORY TO WRITE OUTPUT DATA TO]
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
                           'output_dir' : model_args.output_dir,
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
        self.output_dir = model_args_dict['output_dir']
        self.optics_dir = model_args_dict['optics_dir']
        self.fi_imp = model_args_dict['fi_imp']
    
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
            for i, wvl in enumerate(wvls):
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
                        ssa_ice[i] = ssa_ice_interp(wvl)
                    else:
                        ssa_ice_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                              nearest_ssa_ice[::-1])
                        ssa_ice[i] = ssa_ice_interp(wvl)
                except ValueError:
                    ssa_ice[i] = nearest_ssa_ice[0]
                    sys.stderr.write('error: exception raised while interpolating '
                                     'ssa_ice, using nearest value instead\n')
                    
                # ext_cff_mss_ice
                nearest_ext_cff_mss_ice = ext_in[idx_wvl[:2]]
                try:
                    if nearest_wvls[0] < nearest_wvls[1]:
                        ext_cff_mss_ice_interp = interpolate.interp1d(nearest_wvls,
                                                            nearest_ext_cff_mss_ice)
                        ext_cff_mss_ice[i] = ext_cff_mss_ice_interp(wvl)
                    else:
                        ext_cff_mss_ice_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                      nearest_ext_cff_mss_ice[::-1])
                        ext_cff_mss_ice[i] = ext_cff_mss_ice_interp(wvl)
                except ValueError:
                    ext_cff_mss_ice[i] = nearest_ext_cff_mss_ice[0]
                    sys.stderr.write('error: exception raised while interpolating '
                                     'ext_cff_mss_ice, using nearest value instead\n')
            
                # g
                nearest_g = asm_in[idx_wvl[:2]]
                try:
                    if nearest_wvls[0] < nearest_wvls[1]:
                        g_interp = interpolate.interp1d(nearest_wvls, nearest_g)
                        g[i] = g_interp(wvl)
                    else:
                        g_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                        nearest_g[::-1])
                        g[i] = g_interp(wvl)
                except ValueError:
                    g[i] = nearest_g[0]
                    sys.stderr.write('error: exception raised while interpolating '
                                     'g, using nearest value instead\n')
            
        #snow_optics.close()

        # impurity optics:
        fi_imp = os.path.join(self.optics_dir, self.fi_imp)
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
                sys.stderr.write('error: exception raised while interpolating '
                                 'ssa_imp, using nearest value instead\n')
            
            ssa_imp = np.array([ssa_imp])
            
            # ext_cff_mss_imp
            nearest_ext_cff_mss_imp = ext_in_imp[idx_wvl[:2]]
            try:
                if nearest_wvls[0] < nearest_wvls[1]:
                    ext_cff_mss_imp_interp = interpolate.interp1d(nearest_wvls,
                                                            nearest_ext_cff_mss_imp)
                    ext_cff_mss_imp = ext_cff_mss_imp_interp(wvl)
                else:
                    ext_cff_mss_imp_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                      nearest_ext_cff_mss_imp[::-1])
                    ext_cff_mss_imp = ext_cff_mss_imp_interp(wvl)
            except ValueError:
                ext_cff_mss_imp = nearest_ext_cff_mss_imp[0]
                sys.stderr.write('error: exception raised while interpolating '
                                 'ext_cff_mss_imp, using nearest value instead\n')
            
            ext_cff_mss_imp = np.array([ext_cff_mss_imp])
        
        elif np.size(wvls)>1:
            ssa_imp = np.empty(wvls.shape)
            ext_cff_mss_imp = np.empty(wvls.shape)
            for i, wvl in enumerate(wvls):
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
                        ssa_imp[i] = ssa_imp_interp(wvl)
                    else:
                        ssa_imp_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                              nearest_ssa_imp[::-1])
                        ssa_imp[i] = ssa_imp_interp(wvl)
                except ValueError:
                    ssa_imp[i] = nearest_ssa_imp[0]
                    sys.stderr.write('error: exception raised while interpolating '
                                     'ssa_imp, using nearest value instead\n')
            
                # ext_cff_mss_imp
                nearest_ext_cff_mss_imp = ext_in_imp[idx_wvl[:2]]
                try:
                    if nearest_wvls[0] < nearest_wvls[1]:
                        ext_cff_mss_imp_interp = interpolate.interp1d(nearest_wvls,
                                                            nearest_ext_cff_mss_imp)
                        ext_cff_mss_imp[i] = ext_cff_mss_imp_interp(wvl)
                    else:
                        ext_cff_mss_imp_interp = interpolate.interp1d(nearest_wvls[::-1],
                                                      nearest_ext_cff_mss_imp[::-1])
                        ext_cff_mss_imp[i] = ext_cff_mss_imp_interp(wvl)
                except ValueError:
                    ext_cff_mss_imp[i] = nearest_ext_cff_mss_imp[0]
                    sys.stderr.write('error: exception raised while interpolating '
                                     'ext_cff_mss_imp, using nearest value instead\n')
                    
        #impurity_optics.close()
        
        return(ssa_ice, ext_cff_mss_ice, g, ssa_imp, ext_cff_mss_imp)
    
    def Henyey_Greenstein(self):
        """ Henyey-Greenstein scattering phase function
        """
        costheta_p = self.costheta_p
        g = np.matrix(self.g).T # turn g into column vector
        g_2 = np.multiply(g,g) # compute g^2
        HG_num = 1 - g_2
        HG_den = np.power(1 + g_2 - 2*g*costheta_p, 3./2.)
        p_HG = HG_num / HG_den
        
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
    
    def populate_pdfs(self, g, RANDOM_NUMBERS=1):
        """ 1. Populate PDF of cos(scattering phase angle) with random numbers
            2. Populate PDF of optical path traversed between scattering events
            3. Populate PDF of scattering azimuth angle with random numbers
            4. Populate PDF of single-scatter albedo with random numbers
            5. Populate PDF to determine extinction from ice or impurity
        
            Returns p_rand, tau_rand, phi_rand, ssa_rand, ext_spc_rand
        """
        # 1. Populate PDF of cos(scattering phase angle) with random numbers
        r1 = np.random.rand(RANDOM_NUMBERS) # distribution from 0 -> 1
        p_rand = np.empty((g.size, r1.size))
        tau_rand = np.empty((g.size, r1.size))
        phi_rand = np.empty((g.size, r1.size))
        ssa_rand = np.empty((g.size, r1.size))
        ext_spc_rand = np.empty((g.size, r1.size))
        
        for i, val in enumerate(g):
            p_rand[i, :] = self.Henyey_Greenstein2(val, r1)

            # SANITY CHECK:  mean of the random distribution (should equal g)
            #p_mean = np.mean(p_rand[i,:])
            #print p_mean - val
            
            # 2. Populate PDF of optical path traversed between scattering 
            #    events
            # Q1:  After an optical path of tau, how many photons have NOT 
            #      suffered an extinction event?
            tau_rand[i,:] = -np.log(np.random.rand(RANDOM_NUMBERS))
        
            # median of tau_rand should be -log(0.5)=0.6931
            #tau_median = np.median(tau_rand)
                  
            # 3. Populate PDF of scattering azimuth angle with random numbers
            phi_rand[i,:] = np.random.rand(RANDOM_NUMBERS) * 2*np.pi # 0 -> 2pi
        
            # 4. Populate PDF of single-scatter albedo with random numbers
            ssa_rand[i,:] = np.random.rand(RANDOM_NUMBERS) # 0 -> 1
        
            # 5. Populate PDF to determine extinction from ice or impurity
            ext_spc_rand[i,:] = np.random.rand(RANDOM_NUMBERS) # 0 -> 1
                                   
        return(p_rand, tau_rand, phi_rand, ssa_rand, ext_spc_rand)
    
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
    
    def monte_carlo3D(self, wvl):
        """ Translated from matlab to python by Adam Schneider
        """        
        i_max = self.p_rand.shape[1]
        # initialization:
        x_tau = np.array([0])
        y_tau = np.array([0])
        z_tau = np.array([0])
        
        # initial direction cosines
        mux_0 = 0
        muy_0 = 0
        muz_0 = -1
        
        x_crt = np.array([0])
        y_crt = np.array([0])
        z_crt = np.array([0])
        
        path_length = 0
        
        if self.debug: # debugging / demonstration of 2 scattering events:
            self.x_tau = x_tau
            self.y_tau = y_tau
            self.z_tau = z_tau
            
            self.mux_0 = mux_0
            self.muy_0 = muy_0
            self.muz_0 = muz_0            
            
            # 1. photon enters from above, moving straight down:
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
            mux_0 = 0
            muy_0 = 0
            muz_0 = -1
            
            x_crt = np.array([0])
            y_crt = np.array([0])
            z_crt = np.array([0])
            
            path_length = 0
        
        # scatter the photon inside the cloud/snow until it escapes or is 
        # absorbed
        condition = 0
        i = 0
        i_rand = 0
        while condition==0:
            i+=1
            i_rand+=1
            if i_rand > i_max: # we need more random numbers!
                g = np.array([self.g[self.photon]])
                pdfs = self.populate_pdfs(g)
                self.p_rand[self.photon] = pdfs[0][0]
                self.tau_rand[self.photon] = pdfs[1][0]
                self.phi_rand[self.photon] = pdfs[2][0]
                self.ssa_rand[self.photon] = pdfs[3][0]
                self.ext_spc_rand[self.photon] = pdfs[4][0]
                
                i_rand = 1
            
            # distance, in optical depth space, to move photon
            dtau_current = self.tau_rand[self.photon, i_rand-1]
            if self.Lambertian and i==1: # photon scattered at surface
                dtau_current = 0
            
            # scattering phase angle:
            if i==1: # the photon enters travelling straight down
                costheta = 1
                sintheta = 0
            elif self.Lambertian:
                valid_val=0
                while valid_val==0:
                    theta_rand = np.random.uniform(0, np.pi/2)
                    r1 = np.random.rand()
                    if r1 < 2 * np.sin(theta_rand) * np.cos(theta_rand):
                        costheta = np.cos(theta_rand + np.arccos(muz_0))
                        sintheta = np.sqrt(1 - costheta**2)
                        valid_val=1
            else:
                costheta = self.p_rand[self.photon, i_rand-1]
                sintheta = np.sqrt(1 - costheta**2)
                
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
            
            # update tau coordinates:
            x_tau = np.append(x_tau, x_tau[i-1] + dtau_current*mux_n)
            y_tau = np.append(y_tau, y_tau[i-1] + dtau_current*muy_n)
            z_tau = np.append(z_tau, z_tau[i-1] + dtau_current*muz_n)
                        
            # update Cartesian coordinates:
            ext_cff_mss = self.ext_cff_mss[self.photon]
            ext_cff = ext_cff_mss * self.rho_snw
            x_crt = np.append(x_crt, x_crt[i-1] + 
                              dtau_current * mux_n / ext_cff)                            
            y_crt = np.append(y_crt, y_crt[i-1] + 
                              dtau_current * muy_n / ext_cff)
            z_crt = np.append(z_crt, z_crt[i-1] + 
                              dtau_current * muz_n / ext_cff)
                              
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
                condition = 3
                # correct path_length (we only want photon path length within
                # the snow pack)
                path_length = self.tau_tot / ext_cff
                
            elif z_tau[i] < -self.tau_tot:
                # photon has left the bottom of the cloud/snow (diffuse 
                # transmittance)
                condition = 2
                # correct path_length (we only want photon path length within
                # the snow pack)
                correction = -(((z_tau[i] + self.tau_tot) * dtau_current) /
                               ((z_tau[i] - z_tau[i-1]) * ext_cff))
                path_length += correction
            
            elif self.ssa_rand[self.photon, i_rand-1] >= ssa_event:
                # photon was absorbed, archive which species absorbed it:
                if ext_state==1:
                    condition = 4
                elif ext_state==2:
                    condition = 5

        wvn = 1. / wvl
        theta_n = np.arccos(muz_0)
        phi_n = np.arctan(muy_0 / mux_0)
        n_scat = i-1 # number of scattering events
        
        return(condition, wvn, theta_n, phi_n, n_scat, path_length)
              
    def run(self, n_photon, wvl0, half_width, rds_snw, test=False, debug=False,
            Lambertian=False, Lambertian_reflectance=1.):
        """ Run the Monte Carlo model given a normal distribution of
            wavelengths [um].  This better simulates what NERD does with
            non-monochromatic LEDs.
            
            ALL VALUES IN MICRONS
        """
        self.debug = debug
        self.Lambertian = Lambertian
        if self.Lambertian:
            self.R_Lambertian = Lambertian_reflectance
        # Convert half_width to standard deviation
        scale = half_width / 2.355
        
        # Generate random array of photon wavelengths, rounded to nearest nm
        wvls = np.around(np.random.normal(loc=wvl0, scale=scale,
                                          size=(n_photon)), decimals=3)
        par_wvls = Parallel(wvls)
        
        (ssa_ice,
         ext_cff_mss_ice,
         g, 
         ssa_imp,
         ext_cff_mss_imp) = self.get_optical_properties(par_wvls.working_set,
                                                        rds_snw)
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
        self.costheta_p = np.arange(-1.000, 1.001, 0.001)
        self.g = g
        #self.p = self.Henyey_Greenstein()
        self.ssa_ice = ssa_ice
        self.ssa_imp = ssa_imp
        
        (self.p_rand,
         self.tau_rand,
         self.phi_rand,
         self.ssa_rand,
         self.ext_spc_rand) = self.populate_pdfs(g)
        
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
    
    def plot_phase_function(self):
        """ plot phase function versus cos(theta)
        
            For computational purposes, will only plot the first 100 curves.
            The first 100 curves are random, so this will give a good sample 
            for larger N
        """
        self.p = self.Henyey_Greenstein()
        p = np.asarray(self.p)
        
        fig = plt.figure()
        if np.size(self.g)>1:
            mean_g = np.around(np.mean(self.g), 4)
            std_g = np.around(np.std(self.g), 4)
            for i, val in enumerate(self.g):
                if i < 1000:
                    plt.semilogy(self.costheta_p, p[i])
                    
                    plt.title('Henyey-Greenstein Phase Function for\n'
                              'mean(g) = %s and std(g) = %s' % (mean_g, std_g),
                              fontsize=18)
        elif np.size(self.g)==1:
            g_rounded = np.around(self.g, 4)
            plt.semilogy(self.costheta_p, p[0])
            
            plt.title('Henyey-Greenstein Phase Function for\n'
                      'g = %s' % g_rounded[0], fontsize=18)
        plt.xlabel(r'$\cos(\theta)$', fontsize=18)
        plt.ylabel('Relative probability', fontsize=18)
        plt.xlim((-1, 1))
        plt.grid()
        
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
    test_case.ssa_ice = ssa_ice
    test_case.g = g
    
    test_case.run(n_photon, wvl, half_width, rds_snw, test=True)
    test_case.plot_phase_function()

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
    test_case.ext_cff_mss_ice = ext_cff_mss_ice
    test_case.ssa_ice = ssa_ice
    test_case.g = g
    test_case.ext_cff_mss_imp = ext_cff_mss_imp
    test_case.ssa_imp = ssa_imp
    
    test_case.run(n_photon, wvl, half_width, rds_snw, test=True)
    test_case.plot_phase_function()

def run():
    """ USER INPUT
    """
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
                                   fi_imp=fi_imp)
                                   
    monte_carlo_model.run(n_photon, wvl, half_width, rds_snw)
    monte_carlo_model.plot_phase_function()

def main():
    run()

if __name__=='__main__':
    main()