#!/usr/bin/env python

import os
import fnmatch

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

FIGURE_STYLE = 'agu_half_horizontal'
DATA_DIR = '/data1/amaschne/agu16'
#DATA_DIR = 'data'
COLOR_MAP = 'plasma'
WVL = 1.5
HALF_WIDTH = 1e-12
N_PHOTON = 2000000
THETA_0 = 55.0
STOKES = 1000

class AGU16Data(object):
    def __init__(self, shape, roughness):
        self.data_dir = os.path.join(DATA_DIR, shape, 'Rough%s' % roughness)
        self.shape = shape
    def reduce(self):
        expression = '%s_%s_*_%s_%s_I%s*.txt' % (WVL, HALF_WIDTH, N_PHOTON,
                                                THETA_0, STOKES)
        file_dict = dict()
        for i, fi in enumerate(os.listdir(self.data_dir)):
            if fnmatch.fnmatch(fi, expression):
                RE = fi.split('_')[2]
                file_dict[RE] = list()
        for i, fi in enumerate(os.listdir(self.data_dir)):
            if fnmatch.fnmatch(fi, expression):
                RE = fi.split('_')[2]
                file_dict[RE].append(fi)        
        
        for RE, file_list in file_dict.items():
            total_photons = len(file_list) * N_PHOTON
            f_out = 'R_%s_%s_%s_%s_%s_I%s.txt' % (WVL, HALF_WIDTH, RE,
                                                total_photons, THETA_0, STOKES)
            f_out_path = os.path.join(self.data_dir, f_out)
            
            reflected_list = list()
            for i, fi in enumerate(file_list):
                fi_path = os.path.join(self.data_dir, fi)
                print('reducing %s --> %s' % (fi_path, f_out_path))
                data = pd.read_csv(fi_path, delim_whitespace=True)
                reflected_photons = data[data.condition==1]
                reflected_list.append(reflected_photons)
                
            all_reflected_photons = pd.concat(reflected_list)
            
            all_reflected_photons.to_csv(f_out_path, sep=' ')
    
    def find_data(self):
        expression = 'R_%s_%s_*_%s_I%s.txt' % (WVL, HALF_WIDTH,
                                             THETA_0, STOKES)
        file_dict = dict()
        for i, fi in enumerate(os.listdir(self.data_dir)):
            if fnmatch.fnmatch(fi, expression):
                RE = fi.split('_')[3]
                file_dict[RE] = fi
        
        self.file_dict = file_dict
        
    def contourf_brf(self, theta_bins=9, phi_bins=36, nlevels=8, rmax=.14,
                     zero_loc='S'):
        levels = np.linspace(0, rmax, nlevels)
        self.find_data()
        self.setup_figure1(ncols=3)
        
        theta_range = (0., np.pi/2)
        phi_range = (0., 2*np.pi)
        
        col_num = 0
        ax_list = list()
        for RE, file_name in sorted(self.file_dict.items()):
            ax_list.append(self.ax_arr[col_num])
            RE_int = np.around(float(RE))
            file_path = os.path.join(self.data_dir, file_name)
            
            total_photons = int(file_name.split('_')[4])
            
            data_file = pd.read_csv(file_path, delim_whitespace=True)
            
            theta_exit = data_file['theta_n']
            phi_exit = data_file['phi_n']
            
            h = np.histogram2d(phi_exit, theta_exit, bins=[phi_bins, 
                               theta_bins], range=[phi_range, theta_range])
            phi_midpoints = (np.diff(h[1]) / 2.) + h[1][:-1]
            theta_midpoints = (np.diff(h[2]) / 2.) + h[2][:-1]
            
            theta_weights = (np.sin(theta_midpoints) * np.cos(theta_midpoints)/ 
                     np.sum(np.sin(theta_midpoints) * np.cos(theta_midpoints)))
            
            brf_weights = theta_weights / phi_bins
            brf = (h[0] / (total_photons * brf_weights)).T
            
            print ('max BRF = %r' % brf.max())
            
            theta_deg = np.rad2deg(theta_midpoints)
            phi_rad = np.linspace(0, 2*np.pi, phi_midpoints.size)
            
            self.ax_arr[col_num].set_theta_zero_location(zero_loc)
            cax = self.ax_arr[col_num].contourf(phi_rad, theta_deg,
                                                  brf,
                                                  levels,
                                                  cmap=COLOR_MAP)
        
            #self.ax_arr[col_num].set_title('%d $\mathrm{\mu m}$' % RE_int)
            self.ax_arr[col_num].set_title('%d um' % RE_int)
            col_num +=1
        cb = self.fig.colorbar(cax, orientation='horizontal', 
                               ax=ax_list)
        cb.set_label('Reflectance factor')
                
    def setup_figure1(self, ncols=3, style=FIGURE_STYLE):
        plt.style.use(style)
        self.fig, self.ax_arr = plt.subplots(1, 
                                        ncols,
                                        subplot_kw=dict(projection='polar'))
        self.fig.suptitle('Bidirectional reflectance factors for %s'
                          % self.shape.replace('_',' '))

        return self.fig, self.ax_arr

def main():
    shape = 'column_8elements'
    #shape = 'droxtal'
    roughness = '050'
    fig_dir = os.path.join('/home/amaschne/Projects', 'agu16', 'figs')
    fig_path = os.path.join(fig_dir, '%s.jpg' % shape)
    #fig_path = '%s.jpg' % shape
    
    agu16_data = AGU16Data(shape, roughness)
    agu16_data.contourf_brf()
    plt.savefig(fig_path, dpi=300)
        
if __name__=='__main__':
    main()