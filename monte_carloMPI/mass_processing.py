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
        
        self.fontsize = fontsize
        self.nrows = nrows
        self.ncols = ncols
        self.fig = fig
        self.axarr = axarr
        self.shapes = shapes
        self.roughnesses = roughnesses
        
    def Henyey_Greenstein_phase_function(self, g):
        """
        """
        g2 = g**2
        
        num = 1 - g2
        den = (1 + g2 - 2 * g * self.cos_Theta)**(3./2.)
        
        P_HG = num / den
        
        return P_HG
    
    def full_scattering_phase_function(self):
        """ Calculate the scattering phase function for arbitrarly polarized 
            light
        """
        P_Theta_Phi = self.I * self.P11 + self.P12 * (self.Q * self.cos_2Phi +
                                                      self.U * self.sin_2Phi)
                                                 
        P_Theta = np.mean(P_Theta_Phi, axis=0)
        
        return P_Theta
        
    def plot_phase_functions(self, wvl, rds_snw_list,
                             stokes_params=np.array([1,0,0,0])):
        """
        """
        self.stokes_params = stokes_params
        phase_function = monte_carlo3D.MonteCarlo()
        phase_function.wvl0 = wvl
        wvl = np.array([wvl])
        
        # unpack stokes parameters
        self.I = stokes_params[0]
        self.Q = stokes_params[1]
        self.U = stokes_params[2]
        self.V = stokes_params[3]        
        
        Phi = np.linspace(0, TWO_PI, 36000)
        Phi.shape = (1, 36000)
        Phi = Phi.T
        
        self.cos_2Phi = np.cos(2*Phi)
        self.sin_2Phi = np.sin(2*Phi)
        
        self.cos_Theta_HG = np.linspace(-1, 1, 18000)
        Theta_HG = np.arccos(self.cos_Theta_HG)
        
        colors = ['b','g','r','c','m','y','k']
        for row, roughness in enumerate(self.roughnesses):
            for col, shape in enumerate(self.shapes):
                ax = self.axarr[row, col]
                for i, rds_snw in enumerate(rds_snw_list):
                    print('Working on %d micron %s %s...' % (rds_snw,
                                                             roughness,
                                                             shape))
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
                                                                  
                    try:
                        self.P11 = phase_function.P11[wvl[0]]
                        self.P12 = phase_function.P12[wvl[0]]
                        data_exists = True
                    except KeyError:
                        data_exists = False
                    
                    if data_exists:
                        Theta_P11 = phase_function.theta_P11
                        #self.Theta_P12 = phase_function.theta_P12
                
                        P_Theta = self.full_scattering_phase_function()
                        P_HG = self.Henyey_Greenstein_phase_function(self, g[0])
                        
                        RE = np.around(phase_function.snow_effective_radius)
                        color = colors[i]
                        ax.semilogy(Theta_P11, P_Theta, color=color,
                                    label='%d' % RE)
                                    
                        ax.semilogy(Theta_HG, P_HG, color=color,
                                    linestyle='dashed')
                        plt.grid(True)
                        plt.legend(title='Snow effective radius '
                                         r'($\mathrm{\mu m}$)',
                                   fontsize = self.fontsize )
                                   
def main():
    wvl = 1.3
    rds_snw_list = [50,100,250,500,1000]
    subplots = Subplots()
    subplots.plot_phase_functions(wvl, rds_snw_list)
    plt.show()

if __name__=='__main__':
    main()
