#!/usr/bin/env python

import os
import fnmatch

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import pandas as pd

FIGURE_STYLE = 'agu_half_horizontal'
DATA_DIR = '/data1/amaschne/agu16'
FIG_DIR = os.path.join('/home/amaschne/Projects', 'agu16', 'figs')
#DATA_DIR = 'data'
COLOR_MAP = 'plasma'
WVL = 1.5
HALF_WIDTH = 1e-12
N_PHOTON = 2000000
THETA_0 = 55.0
STOKES = 1000

def main():
    shapes = ['droxtal', 'column_8elements']
    roughnesses = ['000', '050']
    
    for i, shape in enumerate(shapes):
        print('working on BRF for %s' % shape)
        fig_brf(shape=shape, roughness=roughnesses[i])

        print('working on deg. of pol. for %s' % shape)
        fig_deg_pol(shape=shape, roughness=roughnesses[i])
    
def fig_brf(shape, roughness):
    fig_path = os.path.join(FIG_DIR, 'brf_%s.jpg' % shape)
    
    agu16_data = AGU16Data(shape, roughness)
    
    agu16_data.contourf_brf()
    plt.savefig(fig_path, dpi=450)

def fig_deg_pol(shape, roughness):
    fig_path = os.path.join(FIG_DIR, 'deg_pol_%s.jpg' % shape)
    
    agu16_data = AGU16Data(shape, roughness)

    agu16_data.contourf_degree_polarization()
    plt.savefig(fig_path, dpi=450)

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
        
    def contourf_degree_polarization(self, theta_bins=9, phi_bins=36, 
                                     nlevels=51, rmax = 1.0, zero_loc='S'):
        bounds = np.linspace(0,rmax,nlevels)
        norm = colors.BoundaryNorm(bounds, 256)
        
        self.find_data()
        
        shape_title = self.shape.replace('_', ' ')
        fig_title = 'Degree of polarization for %s' % shape_title
        
        self.setup_figure(fig_title, ncols=3)
        
        theta_range = (0., np.pi/2)
        phi_range = (0., 2*np.pi)
        
        theta_deg = np.rad2deg(np.linspace(0, np.pi/2, theta_bins))
        phi_rad = np.linspace(0, 2*np.pi, phi_bins)
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
            
            # Stokes parameters
            Q_exit = data_file['Q_n,']
            U_exit = data_file['U_n,']
            V_exit = data_file['V_n']
            
            h_I = np.histogram2d(phi_exit, theta_exit,
                                 bins=[phi_bins, theta_bins],
                                 range=[phi_range, theta_range])
            h_Q = np.histogram2d(phi_exit, theta_exit,
                                 bins=[phi_bins, theta_bins],
                                 range=[phi_range, theta_range],
                                 weights=Q_exit)
            h_U = np.histogram2d(phi_exit, theta_exit,
                                 bins=[phi_bins, theta_bins],
                                 range=[phi_range, theta_range],
                                 weights=U_exit)
            h_V = np.histogram2d(phi_exit, theta_exit,
                                 bins=[phi_bins, theta_bins],
                                 range=[phi_range, theta_range],
                                 weights=V_exit)
                                 
            Q_norm = h_Q[0] / h_I[0]
            U_norm = h_U[0] / h_I[0]
            V_norm = h_V[0] / h_I[0]
            
            deg_polarization = np.sqrt(Q_norm**2 + U_norm**2 + V_norm**2).T
            phi_edges = h_I[1]
            theta_edges = np.rad2deg(h_I[2])

            print('max deg. of pol. = %r' % deg_polarization.max())
            
            self.ax_arr[col_num].set_theta_zero_location(zero_loc)
            cax = self.ax_arr[col_num].pcolor(phi_edges, theta_edges,
                                                deg_polarization,
                                                 cmap=COLOR_MAP,
                                                 norm=norm)
        
            #self.ax_arr[col_num].set_title('%d $\mathrm{\mu m}$' % RE_int)
            self.ax_arr[col_num].set_title('%d um' % RE_int)
            col_num +=1
        cb = self.fig.colorbar(cax, orientation='horizontal', 
                               ax=ax_list, ticks=np.arange(0,1.1,0.1))
        
    def contourf_brf(self, theta_bins=9, phi_bins=36, nlevels=8, rmax=.14,
                     zero_loc='S'):
        bounds = np.linspace(0, rmax, nlevels)
        norm = colors.BoundaryNorm(bounds, 256)
        
        self.find_data()
            
        shape_title = self.shape.replace('_',' ')
        
        fig_title = 'Bidirectional reflectance factors for %s' % shape_title
        
        self.setup_figure(fig_title, ncols=3)
        
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
            
            h = np.histogram2d(phi_exit, theta_exit,
                               bins=[phi_bins, theta_bins],
                               range=[phi_range, theta_range])
            phi_midpoints = (np.diff(h[1]) / 2.) + h[1][:-1]
            theta_midpoints = (np.diff(h[2]) / 2.) + h[2][:-1]
            
            theta_weights = (np.sin(theta_midpoints) * np.cos(theta_midpoints)/ 
                     np.sum(np.sin(theta_midpoints) * np.cos(theta_midpoints)))
            
            brf_weights = theta_weights / phi_bins
            brf = (h[0] / (total_photons * brf_weights)).T
            
            Phi_rad = h[1]
            Theta_deg = np.rad2deg(h[2])
            
            print ('max BRF = %r' % brf.max())
            
            self.ax_arr[col_num].set_theta_zero_location(zero_loc)
            cax = self.ax_arr[col_num].pcolormesh(Phi_rad, Theta_deg,
                                                  brf,
                                                  cmap=COLOR_MAP,
                                                  norm=norm)
        
            #self.ax_arr[col_num].set_title('%d $\mathrm{\mu m}$' % RE_int)
            self.ax_arr[col_num].set_title('%d um' % RE_int)
            col_num +=1
        cb = self.fig.colorbar(cax, orientation='horizontal', 
                               ax=ax_list)
        cb.set_label('Reflectance factor')
                
    def setup_figure(self, title, ncols=3, style=FIGURE_STYLE):
        plt.style.use(style)
        self.fig, self.ax_arr = plt.subplots(1, 
                                        ncols,
                                        subplot_kw=dict(projection='polar'))
        self.fig.suptitle(title)

        return self.fig, self.ax_arr
        
if __name__=='__main__':
    main()