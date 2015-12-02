#!/usr/bin/env python
"""
"""

import numpy as np
from matplotlib import pyplot as plt

import monte_carlo3D

import ipdb

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
                              'severely rough'])
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
                
        plt.show()
        
        self.nrows = nrows
        self.ncols ncols
        self.fig = fig
        self.axarr = axarr
        self.shapes = shapes
        self.roughnesses = roughnesses
        
    def plot_phase_functions(self, wvl, rds_snw_list,
                             stokes_params=np.array([1,0,0,0])):
        """
        """
        for row, roughness in enumerate(self.roughnesses):
            for col, shape in enumerate(self.shapes):
                for i, rds_snw in enumerate(rds_snw_list):
                    phase_function = monte_carlo3D.MonteCarlo()
                    
                    # get ice optical data
                    if wvl >= 0.2 and wvl <= 15.25:
                        phase_function.far_IR = False
                        (ssa_ice,
                         ext_cff_mss_ice,
                         g) = phase_function.get_aspherical_SSPs([wvl], rds_snw)
                         
                    elif wvl >= 16.4 and wvl <= 99.0:
                         phase_function.far_IR = True
                         (ssa_ice,
                          ext_cff_mss_ice,
                          g) = phase_function.get_aspherical_SSPs([wvl],
                                                                  rds_snw)
                    ipdb.set_trace()
def main():
    subplots = Subplots()
    subplots.plot_phase_functions()

if __name__=='__main__':
    main()