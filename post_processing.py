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

def bi_directional_reflectance_factor(args, file_list):
    """ take list of monte_carlo3D output files and create polar plots of the
        bi-rectional reflectance factors
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
        data = pd.read_csv(file_path, delim_whitespace=True)
        
        data_dict[rds_snw][wvl0] = data

    rds_snw_list = sorted(data_dict.keys())
    
    #fig = plt.figure()
    
    colors = ['blue', 'red']
    hist_range = (0., np.pi/2)
    r_step = (args.r_max / 10) - 0.01
    albedo_dict = dict()
    for i, rds_snw in enumerate(rds_snw_list):
        #plt.subplot(1,len(rds_snw_list),i+1, polar=True)
        fig = plt.figure()
        auxa = polar_demo.fractional_polar_axes(fig, thlim=(0, 90),
                                                rlim=(0,args.r_max),
                                                step=(15, r_step),
                                                thlabel='Elevation angle',
                                                rlabel='Reflectance factor')

        albedo_dict[rds_snw] = dict()
        wvl0_list = sorted(data_dict[rds_snw].keys())
        for j, wvl0 in enumerate(wvl0_list):
            data = data_dict[rds_snw][wvl0]
            
            Q_down = data['wvn[um^-1]'].sum()
            Q_up = data[data.condition==1]['wvn[um^-1]'].sum()
            
            albedo_dict[rds_snw][wvl0] = Q_up / Q_down
            weights = data[data.condition==1]['wvn[um^-1]']
            theta_exit = data[data.condition==1]['theta_n']
            
            h = np.histogram(theta_exit, bins=args.bins, range=hist_range,
                             weights=weights)                 
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
        
        if args.save_figs:
            plt.savefig('bdrf_%s.pdf' % rds_snw)        
        else:
            plt.show()
    
    # histogram of photon path length
    plt.figure()
    for i, wvl0 in enumerate(wvl0_list):
        for j, rds_snw in enumerate(rds_snw_list):
            plt.subplot(len(wvl0_list), len(rds_snw_list),
                        j + 1 + (i*len(rds_snw_list)))
            path_lengths = data_dict[rds_snw][wvl0]['path_length[m]']*100
            path_lengths.hist(bins=args.path_length_bins)
            h = np.histogram(path_lengths, bins=args.path_length_bins)
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
    
    if args.save_figs:        
        plt.savefig('photon_pathlength_hist.pdf')
    else:
        plt.show()

    # histogram of number of scattering events
    plt.figure()
    for i, wvl0 in enumerate(wvl0_list):
        for j, rds_snw in enumerate(rds_snw_list):
            plt.subplot(len(wvl0_list), len(rds_snw_list),
                        j + 1 + (i*len(rds_snw_list)))
            n_scat = data_dict[rds_snw][wvl0]['n_scat']
            n_scat.hist(bins=args.n_scat_bins)
            h = np.histogram(n_scat, bins=args.n_scat_bins)
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
    
    if args.save_figs:        
        plt.savefig('scattering_events_hist.pdf')
    else:
        plt.show()
        
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
        plt.savefig('albedo.pdf')
    else:
        plt.show()
                                 
def get_args():
    config = ConfigParser.SafeConfigParser()
    config_file = 'config.ini'
    config.read(config_file)
    
    section_name = 'data'
    output_dir = config.get(section_name, 'output_dir')
    
    parser = argparse.ArgumentParser(description='[DESCRIPTION]')
    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help='directory containing "monte_carlo3D"')
    parser.add_argument('--bins', type=int, default=90,
                        help='number of theta bins to compute histogram over')
    parser.add_argument('--path_length_bins', type=int, default=1000,
                        help='number of x bins to compute histogram over')
    parser.add_argument('--n_scat_bins', type=int, default=200,
                        help='number of x bins to compute histogram over')
    parser.add_argument('--r_max', type=float, default=1.0)
    parser.add_argument('--save_figs', action='store_true')
    parser.add_argument('file_list', nargs='*')
                        
    args = parser.parse_args()
    
    return args

def process():
    args = get_args()    
    file_list = args.file_list
    
    # Create bi-directional reflectance factor polar plots for each grain size
    if False:
        file_list = ['1.3_0.085_100.0_1000000.txt',
                     '1.3_0.085_300.0_1000000.txt',
                     '1.3_0.085_500.0_1000000.txt',
                     '1.3_0.085_700.0_1000000.txt',
                     '1.3_0.085_900.0_1000000.txt',
                     '1.55_0.085_100.0_1000000.txt',
                     '1.55_0.085_300.0_1000000.txt', 
                     '1.55_0.085_500.0_1000000.txt',
                     '1.55_0.085_700.0_1000000.txt', 
                     '1.55_0.085_900.0_1000000.txt']
    if False: # only wvl0 = 1.55 um
        file_list = ['1.55_0.085_100.0_1000000.txt',
                     '1.55_0.085_300.0_1000000.txt', 
                     '1.55_0.085_500.0_1000000.txt',
                     '1.55_0.085_700.0_1000000.txt', 
                     '1.55_0.085_900.0_1000000.txt']          
    
    bi_directional_reflectance_factor(args, file_list)
                                
def main():
    process()

if __name__=='__main__':
    main()