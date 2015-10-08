#!/usr/bin/env python

""" process and plot results from monte_carlo3D
"""

import os
import sys

import ConfigParser
import argparse

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import polar_demo

import ipdb as pdb

class MonteCarloData(object):
    def __init__(self, args, file_list):
        data_dir = args.output_dir
    
        rds_snw_list = list()
        for i, name in enumerate(file_list):
            rds_snw_list.append(float(name.split('_')[2]))
    
        data_dict = dict()
        for i, rds_snw in enumerate(rds_snw_list):
            data_dict[rds_snw] = dict()
    
        for i, name in enumerate(file_list):
            wvl0 = float(name.split('_')[0])
            half_width = float(name.split('_')[1])
            rds_snw = float(name.split('_')[2])
        
            file_path = os.path.join(data_dir, name)
            data = pd.read_csv(file_path, delim_whitespace=True)
        
            data_dict[rds_snw][wvl0] = data
            
        self.args = args
        self.rds_snw_list = sorted(data_dict.keys())
        self.data_dict = data_dict
    

    def bi_directional_reflectance_factor(self):
        """ take list of monte_carlo3D output files and create polar plots of 
            the bi-rectional reflectance factors
        """
        #fig = plt.figure()
    
        colors = ['blue', 'red']
        hist_range = (0., np.pi/2)
        r_step = (self.args.r_max / 10) - 0.01
        albedo_dict = dict()
        for i, rds_snw in enumerate(self.rds_snw_list):
            #plt.subplot(1,len(rds_snw_list),i+1, polar=True)
            fig = plt.figure()
            auxa = polar_demo.fractional_polar_axes(fig, thlim=(0, 90),
                                                    rlim=(0,self.args.r_max),
                                                    step=(15, r_step),
                                                    thlabel='Elevation angle',
                                                    rlabel='Reflectance factor')

            albedo_dict[rds_snw] = dict()
            wvl0_list = sorted(self.data_dict[rds_snw].keys())
            for j, wvl0 in enumerate(wvl0_list):
                data = self.data_dict[rds_snw][wvl0]
            
                Q_down = data['wvn[um^-1]'].sum()
                Q_up = data[data.condition==1]['wvn[um^-1]'].sum()
            
                albedo_dict[rds_snw][wvl0] = Q_up / Q_down
                weights = data[data.condition==1]['wvn[um^-1]']
                theta_exit = data[data.condition==1]['theta_n']
            
                h = np.histogram(theta_exit, bins=self.args.bins,
                                 range=hist_range, weights=weights)                 
                '''
                plt.hist(theta_exit.values, bins=args.bins, range=hist_range,
                         weights=weights.values, label=str(rds_snw))
                '''
                midpoints = (np.diff(h[1]) / 2.) + h[1][:-1]
                bdrf_weights = (np.sin(midpoints)*np.cos(midpoints) / 
                                np.sum(np.sin(midpoints)*np.cos(midpoints)))
                            
                bdrf = h[0] / (Q_down * bdrf_weights)
                theta_deg = np.rad2deg(midpoints[::-1])    
                auxa.plot(theta_deg, bdrf, label='%s ' % (wvl0) + r'$\mathrm{\mu m}$',
                          color=colors[j])
            
                #plt.polar(midpoints, bdrf, label='%s' % wvl0)
            #snow_radius = np.around(rds_snw)
            #plt.title('Snow radius = %s microns' % snow_radius)    
            plt.legend(bbox_to_anchor=(1, 1),
                       bbox_transform=plt.gcf().transFigure)   
        
            if self.args.save_figs:
                fig_path = os.path.join(self.args.save_dir,
                                        'bdrf_%s.%s' % (rds_snw,
                                                        self.args.fig_format))
                plt.savefig(fig_path)        
            else:
                plt.show()
            plt.close()
    
        if not self.args.no_histograms:
            # histogram of photon path length
            plt.figure()
            for i, wvl0 in enumerate(wvl0_list):
                for j, rds_snw in enumerate(self.rds_snw_list):
                    plt.subplot(len(wvl0_list), len(self.rds_snw_list),
                                j + 1 + (i*len(self.rds_snw_list)))
                    path_lengths = self.data_dict[rds_snw][wvl0]['path_length[m]']*100
                    path_lengths.hist(bins=self.args.path_length_bins)
                    h = np.histogram(path_lengths, bins=self.args.path_length_bins)
                    y_max = np.max(h[0])
                    xticks = np.arange(0, 25, 10)
                    yticks = np.array([])
                    plt.xlim(0, 20)
                    plt.ylim(0, y_max)
                    plt.xticks(xticks)
                    plt.yticks(yticks)
                    if i==0:
                        plt.title('%s ' % (rds_snw) + r'$\mathrm{\mu m}$')
                    elif i==1 and j==2:
                        plt.xlabel('Photon path length [cm]')
            
                    if j==0:
                        plt.ylabel('%s ' % (wvl0) + r'$\mathrm{\mu m}$')
    
            if self.args.save_figs:        
                fig_path = os.path.join(self.args.save_dir, 'photon_pathlength_hist.pdf')
                plt.savefig(fig_path)
            else:
                plt.show()
            plt.close()

            # histogram of number of scattering events
            plt.figure()
            for i, wvl0 in enumerate(wvl0_list):
                for j, rds_snw in enumerate(self.rds_snw_list):
                    plt.subplot(len(wvl0_list), len(self.rds_snw_list),
                                j + 1 + (i*len(self.rds_snw_list)))
                    n_scat = self.data_dict[rds_snw][wvl0]['n_scat']
                    n_scat.hist(bins=self.args.n_scat_bins)
                    h = np.histogram(n_scat, bins=self.args.n_scat_bins)
                    y_max = np.max(h[0])
                    xticks = np.arange(0, 220, 100)
                    yticks = np.array([])
                    plt.xlim(0, 200)
                    plt.ylim(0, y_max)
                    plt.xticks(xticks)
                    plt.yticks(yticks)
                    if i==0:
                        plt.title('%s ' % (rds_snw) + r'$\mathrm{\mu m}$')
                    elif i==1 and j==2:
                        plt.xlabel('Scattering events')
            
                    if j==0:
                        plt.ylabel('%s ' % (wvl0) + r'$\mathrm{\mu m}$')
    
            if self.args.save_figs:        
                fig_path = os.path.join(self.args.save_dir, 'scattering_events_hist.pdf')
                plt.savefig(fig_path)
            else:
                plt.show()
            plt.close()
    
        '''    
        # plot albedo
        plt.figure()
        for i, wvl0 in enumerate(wvl0_list):
            albedo = np.empty((len(rds_snw_list)))
            for j, rds_snw in enumerate(rds_snw_list):
                albedo[j] = albedo_dict[rds_snw][wvl0]

            plt.plot(rds_snw_list, albedo, label='%s ' % (wvl0) + r'$\mathrm{\mu m}$',
                     color=colors[i])
    
        plt.legend()
        plt.grid()
        plt.xlabel(r'Snow radius [$\mathrm{\mu m}$]')
        plt.ylabel('Reflectance')
        plt.title('Nadir directional-hemispherical reflectance')
    
        if args.save_figs:
            fig_path = os.path.join(args.save_dir, 'albedo.pdf')        
            plt.savefig(fig_path)
        else:
            plt.show()
        plt.close()
    ''' 
                                 
    def spectral_albedo(self):
        """ Just read data into workspace - useful for interactive work
        """    
        albedo_dict = dict()
        albedo_dict2 = dict()
        for i, rds_snw in enumerate(self.rds_snw_list):
            albedo_dict[rds_snw] = dict()
            albedo_dict2[rds_snw] = dict()
            wvl0_list = sorted(self.data_dict[rds_snw].keys())
            for j, wvl0 in enumerate(wvl0_list):
                data = self.data_dict[rds_snw][wvl0]
            
                Q_down = data['wvn[um^-1]'].sum()
                Q_up = data[data.condition==1]['wvn[um^-1]'].sum()
                
                # Include photons leaving bottom of snow pack
                Q_non_absorbed = (Q_up + 
                                  data[data.condition==2]['wvn[um^-1]'].sum() +
                                  data[data.condition==3]['wvn[um^-1]'].sum())
            
                albedo_dict[rds_snw][wvl0] = Q_up / Q_down
                albedo_dict2[rds_snw][wvl0] = Q_non_absorbed / Q_down
    
        print(' ')
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, rds_snw in enumerate(self.rds_snw_list):
            albedo = np.empty((len(wvl0_list)))
            albedo2 = np.empty((len(wvl0_list)))
            for j, wvl0 in enumerate(wvl0_list):
                albedo[j] = albedo_dict[rds_snw][wvl0]
                albedo2[j] = albedo_dict2[rds_snw][wvl0]
                
                print('albedo(%s, %s) = %r' % (wvl0, rds_snw, albedo[j]))
            
            plt.plot(wvl0_list, albedo,
                     label='%s ' % (rds_snw) + r'$\mathrm{\mu m}$', color=colors[i])
            plt.plot(wvl0_list, albedo2, color=colors[i])
        plt.legend(title='Snow grain effective radius')
        plt.grid()
        plt.xlim((np.min(wvl0_list), np.max(wvl0_list)))
        plt.xticks(np.arange(np.min(wvl0_list), np.max(wvl0_list) + 0.1, 0.2))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel(r'Wavelength ($\mathrm{\mu m}$)')
        plt.ylabel('Reflectance')
        plt.title('Nadir directional-hemispherical reflectance')
    
        if self.args.save_figs:
            fig_path = os.path.join(args.save_dir, 'albedo.pdf')        
            plt.savefig(fig_path)
        else:
            plt.show()
        plt.close()

        print(' ')

def get_args():
    config = ConfigParser.SafeConfigParser()
    config_file = 'config.ini'
    config.read(config_file)
    
    section_name = 'data'
    output_dir = config.get(section_name, 'output_dir')
    
    parser = argparse.ArgumentParser(description='Syntax:  $ python '
             'post_processing2.py --output_dir [OUTPUT_DIR] '
             '$(ls [OUTPUT_DIR]/[FILE_SUBSET] | xargs)')
    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help='directory containing data')
    parser.add_argument('--bins', type=int, default=90,
                        help='number of theta bins to compute histogram over')
    parser.add_argument('--path_length_bins', type=int, default=1000,
                        help='number of x bins to compute histogram over')
    parser.add_argument('--n_scat_bins', type=int, default=200,
                        help='number of x bins to compute histogram over')
    parser.add_argument('--r_max', type=float, default=1.0)
    parser.add_argument('--save_figs', action='store_true')
    parser.add_argument('file_list', nargs=argparse.REMAINDER)
    parser.add_argument('--save_dir', type=str, default='figures')
    parser.add_argument('--no_histograms', action='store_true')
    parser.add_argument('--fig_format', type=str, default='pdf')
                        
    args = parser.parse_args()
    
    return args

def read_data(args, file_list):
    """ Just read data into workspace - useful for interactive work
    """
    data_dir = args.output_dir
    
    rds_snw_list = list()
    for i, name in enumerate(file_list):
        rds_snw_list.append(float(name.split('_')[2]))    
    
    data_dict = dict()
    for i, rds_snw in enumerate(rds_snw_list):
        data_dict[rds_snw] = dict()
    
    for i, name in enumerate(file_list):
        wvl0 = float(name.split('_')[0])
        half_width = float(name.split('_')[1])
        rds_snw = float(name.split('_')[2])
        
        file_path = os.path.join(data_dir, name)
        print file_path
        data = pd.read_csv(file_path, delim_whitespace=True)
        
        data_dict[rds_snw][wvl0] = data
    
    rds_snw_list = sorted(data_dict.keys())
    
    albedo_dict = dict()
    for i, rds_snw in enumerate(rds_snw_list):
        albedo_dict[rds_snw] = dict()
        wvl0_list = sorted(data_dict[rds_snw].keys())
        for j, wvl0 in enumerate(wvl0_list):
            data = data_dict[rds_snw][wvl0]
            
            Q_down = data['wvn[um^-1]'].sum()
            Q_up = data[data.condition==1]['wvn[um^-1]'].sum()
            
            albedo_dict[rds_snw][wvl0] = Q_up / Q_down
    
    print(' ')
    colors = ['blue', 'red']
    for i, wvl0 in enumerate(wvl0_list):
        albedo = np.empty((len(rds_snw_list)))
        for j, rds_snw in enumerate(rds_snw_list):
            albedo[j] = albedo_dict[rds_snw][wvl0]
            print('albedo(%s, %s) = %r' % (wvl0, rds_snw, albedo[j]))
            
        plt.plot(rds_snw_list, albedo,
                 label='%s ' % (wvl0) + r'$\mathrm{\mu m}$', color=colors[i])
    plt.legend()
    plt.grid()
    plt.xlabel(r'Snow radius [$\mathrm{\mu m}$]')
    plt.ylabel('Reflectance')
    plt.title('Nadir directional-hemispherical reflectance')
    
    if args.save_figs:
        fig_path = os.path.join(args.save_dir, 'albedo.pdf')        
        plt.savefig(fig_path)
    else:
        plt.show()
    plt.close()

    print(' ')
    for i, wvl0 in enumerate(wvl0_list):
        for j, rds_snw in enumerate(rds_snw_list):
            mean_n_scat = data_dict[rds_snw][wvl0]['n_scat'].mean()
            print('mean_n_scat(%s, %s) = %r' % (wvl0, rds_snw, mean_n_scat))

    print(' ')
    for i, wvl0 in enumerate(wvl0_list):
        for j, rds_snw in enumerate(rds_snw_list):
            mean_pathlength = data_dict[rds_snw][wvl0]['path_length[m]'].mean()
            print('mean_pathlength(%s, %s) = %r m' % (wvl0, rds_snw, 
                                                      mean_pathlength))
    
    return data_dict

def process():
    args = get_args()    
    file_list = args.file_list
    for i, file_name in enumerate(file_list):
        file_list[i] = file_name.split('/')[-1]
    
    # Create bi-directional reflectance factor polar plots for each grain size
    monte_carlo_data = MonteCarloData(args, file_list)
    monte_carlo_data.spectral_albedo()
    #monte_carlo_data.bi_directional_reflectance_factor()
    #data_dict = read_data(args, file_list)
    #return data_dict
    
def main():
    process()

if __name__=='__main__':
    main()