#!/usr/bin/env python
"""
"""

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

import monte_carlo3D

import ipdb

PI = np.pi
TWO_PI = 2*np.pi

class Subplots(object):
    def __init__(self, 
                 shapes=['solid hexagonal column', 
                         'hexagonal plate',
                         'hollow hexagonal column',
                         'droxtal',
                         'hollow bullet rosette',
                         'solid bullet rosette',
                         '8-element column aggregate',
                         '5-element plate aggregate',
                         '10-element plate aggreaget'],
                 roughnesses=['smooth',
                              'moderatley rough',
                              'severely rough']):
        nrows = len(roughnesses)
        ncols = len(shapes)
        fig, axarr = plt.subplots(nrows, ncols,sharex='col', sharey='row')
        fontsize = 40./ncols
        for row, roughness in enumerate(roughnesses):
            for col, shape in enumerate(shapes):
                if row == 0:
                    axarr[row,col].set_title('%s' % shape, fontsize=fontsize)
                if col ==0:
                    axarr[row,col].set_ylabel('%s' % roughness, 
                                              fontsize=fontsize)
                
        #plt.show()
        
        self.nrows = nrows
        self.ncols = ncols
        self.fig = fig
        self.axarr = axarr
        self.shapes = shapes
        self.roughnesses = roughnesses
        
    def full_scattering_phase_function(self):
        """ Calculate the scattering phase function for arbitrarly polarized 
            light
        """
        # unpack stokes parameters
        I = self.stokes_params[0]
        Q = self.stokes_params[1]
        U = self.stokes_params[2]
        V = self.stokes_params[3]
        
        two_len_Theta = 2 * self.Theta_P11.size
        Phi = np.linspace(0, TWO_PI, two_len_Theta)
        Phi.shape = (1, two_len_Theta)
        Phi = Phi.T
        
        P_Theta_Phi = I * self.P11 + self.P12 * (Q * np.cos(2*Phi) +
                                                 U * np.sin(2*Phi))
                                                 
        ipdb.set_trace()
        
        
    
    def plot_phase_functions(self, wvl, rds_snw_list,
                             stokes_params=np.array([1,0,0,0])):
        """
        """
        self.stokes_params = stokes_params
        phase_function = monte_carlo3D.MonteCarlo()
        phase_function.wvl0 = wvl
        wvl = np.array([wvl])
        for row, roughness in enumerate(self.roughnesses):
            for col, shape in enumerate(self.shapes):
                for i, rds_snw in enumerate(rds_snw_list):
                    print('Working on %d micron %s %s...' % (rds_snw, roughness, shape))
                    phase_function.roughness = roughness
                    phase_function.shape = shape
                    
                    # get ice optical data
                    if wvl >= 0.2 and wvl <= 15.25:
                        phase_function.far_IR = False
                        (ssa_ice,
                         ext_cff_mss_ice,
                         g) = phase_function.get_aspherical_SSPs(wvl, rds_snw)
                         
                    elif wvl >= 16.4 and wvl <= 99.0:
                         phase_function.far_IR = True
                         (ssa_ice,
                          ext_cff_mss_ice,
                          g) = phase_function.get_aspherical_SSPs([wvl],
                                                                  rds_snw)
                                                                  
                    self.Theta_P11 = phase_function.theta_P11
                    self.Theta_P12 = phase_function.theta_P12
                    self.P11 = phase_function.P11[wvl[0]]
                    self.P12 = phase_function.P12[wvl[0]]
                    self.full_scattering_phase_function()
                    
def main():
    wvl = 1.3
    rds_snw_list = [50,100,250,500,1000]
    subplots = Subplots()
    subplots.plot_phase_functions(wvl, rds_snw_list)

if __name__=='__main__':
    main()
