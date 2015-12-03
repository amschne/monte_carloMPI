#!/usr/bin/env python
"""
"""

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

import monte_carlo3D
from parallelize import Parallel

#import line_profiler
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
                         '10-element plate aggregate'],
                 roughnesses=['smooth',
                              'moderatley rough',
                              'severely rough']):
        nrows = len(roughnesses)
        ncols = len(shapes)
        fig, axarr = plt.subplots(nrows, ncols,sharex='col', sharey='row')
        fontsize = 40./ncols
        row_col_list = list()
        for row, roughness in enumerate(roughnesses):
            for col, shape in enumerate(shapes):
                if row == 0:
                    axarr[row,col].set_title('%s' % shape, fontsize=fontsize)
                if col ==0:
                    axarr[row,col].set_ylabel('%s' % roughness, 
                                              fontsize=fontsize)
                row_col_list.append((row,col))
        self.par_row_cols = Parallel(row_col_list)
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
        den = (1 + g2 - 2 * g * self.cos_Theta_HG)**(3./2.)
        
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
        
    #@profile
    def plot_phase_functions(self, wvl, rds_snw_list,
                             stokes_params=np.array([1,0,0,0])):
        """
        """
        nrds = len(rds_snw_list)
        
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
        len_Theta_HG = Theta_HG.size
        Theta_HG_deg = np.rad2deg(Theta_HG)
        
        answer = list()
        for i, row_col in enumerate(self.par_row_cols.working_set):
            phase_function.roughness = self.roughnesses[row_col[0]]
            phase_function.shape = self.shapes[row_col[1]]
            for k, rds_snw in enumerate(rds_snw_list):
                print('Proc. %d working on %d micron %s %s'
                      % (self.par_row_cols.rank,
                         rds_snw,
                         phase_function.roughness,
                         phase_function.shape))
                    
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
                     g) = phase_function.get_aspherical_SSPs([wvl], rds_snw)
                                                                  
                try:
                    self.P11 = phase_function.P11[wvl[0]]
                    self.P12 = phase_function.P12[wvl[0]]
                    data_exists = True
                except KeyError:
                    data_exists = False
                    
                if data_exists:
                    Theta_P11 = phase_function.theta_P11
                    len_Theta = Theta_P11.size
                    Theta_P11_deg = np.rad2deg(Theta_P11)
                    #self.Theta_P12 = phase_function.theta_P12
                
                    P_Theta = self.full_scattering_phase_function()
                    P_HG = self.Henyey_Greenstein_phase_function(g[0])
                        
                    RE = phase_function.snow_effective_radius
                else:
                    Theta_P11 = None
                    Theta_P11_deg = None
                        
                    P_Theta = None
                    P_HG = None
                        
                    RE = None
                        
                ijk = (row_col[0], row_col[1], k)
                answer.append((ijk, data_exists, Theta_P11_deg,
                               P_Theta, P_HG, RE, g[0], ssa_ice[0]))
                                       
        all_answers = self.par_row_cols.answer_and_reduce(answer,
                                                          Subplots.flatten_list)
        if all_answers is not None:
            # this is the root proocessor
            data_exists = np.empty((self.nrows, self.ncols, nrds))
            Theta_P11_deg = np.empty((self.nrows, self.ncols, nrds, len_Theta))
            P_Theta = np.empty((self.nrows, self.ncols, nrds, len_Theta))
            P_HG = np.empty((self.nrows, self.ncols, nrds, len_Theta_HG))
            RE = np.empty((self.nrows, self.ncols, nrds))
            g = np.empty((self.nrows, self.ncols, nrds))
            ssa_ice = np.empty((self.nrows, self.ncols, nrds))
            
            for num, answer in enumerate(all_answers):
                i = answer[0][0]
                j = answer[0][1]
                k = answer[0][2]
                
                data_exists[i,j,k] = answer[1]
                Theta_P11_deg[i,j,k] = answer[2]
                P_Theta[i,j,k] = answer[3]
                P_HG[i,j,k] = answer[4]
                RE[i,j,k] = answer[5]
                g[i,j,k] = answer[6]
                ssa_ice[i,j,k] = answer[7]
        
            colors = ['b','g','r','c','m','y','k']
            for row, roughness in enumerate(self.roughnesses):
                for col, shape in enumerate(self.shapes):
                    ax = self.axarr[row, col]
                    for k, rds_snw in enumerate(rds_snw_list):
                        if data_exists[row,col,k]:
                            color = colors[k]
                            ax.semilogy(Theta_P11_deg[row,col,k],
                                        P_Theta[row,col,k] * ssa_ice[row,col,k],
                                        color=color,
                                        label='RE = %d $\mathrm{\mu m}$'
                                              % np.around(RE[row,col,k]))
                                    
                            ax.semilogy(Theta_HG_deg,
                                        P_HG[row,col,k] * ssa_ice[row,col,k],
                                        color=color,
                                        linestyle='dashed',
                                        label='HG(g=%s)'
                                              % np.around(g[row,col,k],
                                                          decimals=2))
                    ax.legend(loc=1, fontsize = self.fontsize)
                        
                    ax.set_xticks([0,90,180])
                    ax.set_xlim((-9,189))
                    ax.set_xticklabels([0, 90, 180])
                    ax.grid()
                    
                    if row == self.nrows - 1:
                        # last row
                        ax.set_xlabel(r'$\Theta$')
            plt.show()  

    @classmethod
    def flatten_list(klass, l):
        return [item for sublist in l for item in sublist]

def subsample():
    wvl = 1.3
    rds_snw_list = [50, 250, 1000]
    
    shapes = ['solid hexagonal column',
              'hollow bullet rosette',
              '10-element plate aggregate']
    roughnesses = ['smooth', 'severely rough']
    subplots = Subplots(shapes=shapes, roughnesses=roughnesses)
    subplots.plot_phase_functions(wvl, rds_snw_list)
                                   
def all_phase_funcs():
    wvl = 1.3
    rds_snw_list = [50,100,250,500,1000]
    subplots = Subplots()
    subplots.plot_phase_functions(wvl, rds_snw_list)
    
def main():
    all_phase_funcs()
    #subsample()
    
if __name__=='__main__':
    main()