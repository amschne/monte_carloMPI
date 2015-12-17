#!/usr/bin/env python
"""
"""
import os
import argparse
import fnmatch
import re

import natsort

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import ipdb

import polar_demo

def get_args():
    """ User input
    """
    top_data_dir = '/data3/amaschne/AGU2015'
    shapes = ['sphere',
              'droxtal',
              'solid_hexagonal_column', 
              'hollow_hexagonal_column',
              '8-element_column_aggregate',
              'hexagonal_plate',
              '5-element_plate_aggregate',
              '10-element_plate_aggregate',
              'solid_bullet_rosette',
              'hollow_bullet_rosette']
    roughnesses = ['smooth', 'moderately_rough','severely_rough']
    wvl = 1.3 # um
    half_width = 0.085 # um
    n_photon = 1000000
    theta_0 = 0.0
    Stokes_0 = [1, 0, 0, 0]
    
    """ End user input
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--top_data_dir', default=top_data_dir)
    parser.add_argument('--shapes', nargs='*', default=shapes)
    parser.add_argument('--roughnesses', nargs='*', default=roughnesses)
    parser.add_argument('--wvl', default=wvl)
    parser.add_argument('--half_width', default=half_width)
    parser.add_argument('--n_photon', default=n_photon)
    parser.add_argument('--theta_0', default=theta_0)
    parser.add_argument('--Stokes_0', nargs=4, default=Stokes_0)
    
    args = parser.parse_args()
    args_dict = {'top_data_dir': args.top_data_dir,
                 'shapes': args.shapes,
                 'roughnesses': args.roughnesses,
                 'wvl': args.wvl,
                 'half_width': args.half_width,
                 'n_photon': args.n_photon,
                 'theta_0': args.theta_0,
                 'Stokes_0': args.Stokes_0}
    
    return args_dict

class MonteCarloDataSet(object):
    def __init__(self, **kwargs):
        args_dict = get_args()
        for key, val in kwargs.items():
            args_dict[key] = val
        self.args = args_dict
        
        self.get_data()
    
    def get_data(self):    
        self.data_I = dict()
        self.data_HG = dict()
        for i, shape in enumerate(self.args['shapes']): 
            self.data_I[shape] = dict()
            self.data_HG[shape] = dict()
            
            if shape == 'sphere':
                shape_dir = 'sphere'
            if shape == 'solid_hexagonal_column':
                shape_dir = 'solid_column'
            elif shape == 'hollow_hexagonal_column':
                shape_dir = 'hollow_column'
            elif shape == 'hexagonal_plate':
                shape_dir = 'plate'
            elif shape == 'droxtal':
                shape_dir = 'droxtal'
            elif shape == 'hollow_bullet_rosette':
                shape_dir = 'hollow_bullet_rosette'
            elif shape == 'solid_bullet_rosette':
                shape_dir = 'solid_bullet_rosette'
            elif shape == '8-element_column_aggregate':
                shape_dir = 'column_8elements'
            elif shape == '5-element_plate_aggregate':
                shape_dir = 'plate_5elements'
            elif shape =='10-element_plate_aggregate':
                shape_dir = 'plate_10elements'

            if shape == 'sphere':
                data_dir = os.path.join(self.args['top_data_dir'], shape_dir)
                if os.path.isdir(data_dir):
                    files_I, files_HG = self.get_files(data_dir)
                
                    for k, file_tuple in enumerate(files_I):
                        file_name = file_tuple[0]
                        RE = file_tuple[1]
                        self.data_I[shape][RE] = os.path.join(data_dir,
                                                              file_name)
                
                    for k, file_tuple in enumerate(files_HG):
                        file_name = file_tuple[0]
                        RE = file_tuple[1]
                        self.data_HG[shape][RE] = os.path.join(data_dir,
                                                               file_name)
            
            else:    
                for j, roughness in enumerate(self.args['roughnesses']):
                    if roughness == 'smooth':
                        roughness_dir = 'Rough000'
                    elif roughness == 'moderately_rough':
                        roughness_dir = 'Rough003'
                    elif roughness == 'severely_rough':
                        roughness_dir = 'Rough050'
                    
                    data_dir = os.path.join(self.args['top_data_dir'],
                                            shape_dir,
                                            roughness_dir)
                    if os.path.isdir(data_dir):
                        self.data_I[shape][roughness] = dict()
                        self.data_HG[shape][roughness] = dict()
                    
                        files_I, files_HG = self.get_files(data_dir)
                    
                        for k, file_tuple in enumerate(files_I):
                            file_name = file_tuple[0]
                            RE = file_tuple[1]
                            self.data_I[shape][roughness][RE] = os.path.join(
                                                                      data_dir,
                                                                      file_name)
                    
                        for k, file_tuple in enumerate(files_HG):
                            file_name = file_tuple[0]
                            RE = file_tuple[1]
                            self.data_HG[shape][roughness][RE] = os.path.join(
                                                                      data_dir,
                                                                      file_name)

    def get_files(self, data_dir):
        """ Find Monte Carlo output files in a given directory
        
            Returns lists of file names containing particle effective
            radii
        """
        expression_I = '%s_%s_*_%s_%s_I%s%s%s%s.txt' % (
                                                       self.args['wvl'],
                                                       self.args['half_width'],
                                                       self.args['n_photon'],
                                                       self.args['theta_0'],
                                                       self.args['Stokes_0'][0],
                                                       self.args['Stokes_0'][1],
                                                       self.args['Stokes_0'][2],
                                                       self.args['Stokes_0'][3])
        expression_HG = '%s_%s_*_%s_%s_HG.txt' % (self.args['wvl'],
                                                  self.args['half_width'],
                                                  self.args['n_photon'],
                                                  self.args['theta_0'])
        files_I = list()
        files_HG = list()
        for i, file in enumerate(os.listdir(data_dir)):
            if fnmatch.fnmatch(file, expression_I):
                RE = file.split('_')[2]
                files_I.append((file, RE))
            elif fnmatch.fnmatch(file, expression_HG):
                RE = file.split('_')[2]
                files_HG.append((file, RE))
                
        return(files_I, files_HG)
    
    def plot_brf_all_angles(self,
                            shapes=list(),
                            roughnesses=list(),
                            active_area=1.,
                            d_dome=175.,
                            r_max=1.,
                            r_step=None,
                            savefigs=False):
        """ Plot BRFs for different grain sizes on one plot for a list of given
            shapes and roughnesses.
        """
        wvl_nm = np.around(float(self.args['wvl']) * 10**3)
        zenith = np.around(float(self.args['theta_0']))
        
        colors = ['b','g','r','c','m']
        hist_range = (0., np.pi/2)
        n_bins = calculate_bins(active_area, d_dome)
        
        if len(shapes)==0:
            shapes=self.args['shapes']
        if len(roughnesses)==0:
            roughnesses=self.args['roughnesses']
        
        if r_step is None:
            r_step = (r_max / 10.) - 0.01
        
        for i, shape in enumerate(shapes):
            shape_label = re.sub(r'[\W_]', ' ', shape)
            
            if shape=='sphere':
                fig = plt.figure()
                auxa = polar_demo.fractional_polar_axes(
                            fig, thlim=(0,90), rlim=(0,r_max), step=(15,r_step),
                            thlabel='Elevation angle',
                            rlabel='Reflectance factor')
                
                # Full scattering phase functions
                particle_radii = list()
                theta_deg = list()
                brfs = list()
                for RE, file_path in self.data_I[shape].items():
                    particle_radii.append(np.around(float(RE)))
                    (brf,
                     theta_rad,
                     mean_wvls) = self.bi_directional_reflectance_factor(
                                                                     file_path, 
                                                                     n_bins)
                    theta_deg.append(np.rad2deg(theta_rad))
                    brfs.append(brf)
                
                idxs = np.argsort(particle_radii)
                
                for k, idx in enumerate(idxs):
                    auxa.plot(theta_deg[idx][::-1], brfs[idx],
                              color=colors[k])
                
                # Henyey Greenstein scattering phase functions
                particle_radii = list()
                theta_deg = list()
                brfs = list()
                for RE, file_path in self.data_HG[shape].items():
                    particle_radii.append(np.around(float(RE)))
                    (brf,
                     theta_rad,
                     mean_wvls) = self.bi_directional_reflectance_factor(
                                                                     file_path, 
                                                                     n_bins)
                    theta_deg.append(np.rad2deg(theta_rad))
                    brfs.append(brf)
                
                idxs = np.argsort(particle_radii)
                
                for k, idx in enumerate(idxs):
                    auxa.plot(theta_deg[idx][::-1], brfs[idx],
                              label='%d' % particle_radii[idx], 
                              color=colors[k], linestyle='dashed')
                              
                #plt.title('%dnm %d$^{\circ}$ BRFs of ice %ss' % (wvl_nm,
                 #                                                zenith,
                  #                                               shape_label))
                plt.title('%ss' % (shape_label))
                plt.legend(bbox_to_anchor=(1, 1),
                           bbox_transform=plt.gcf().transFigure,
                           title='RE ($\mathrm{\mu m}$)',
                           fontsize='small')
                           
                if savefigs:
                    figname = '%s_%dbrf.pdf' % (shape, zenith)
                    plt.savefig(figname)
                else:
                    plt.show()
                    plt.close()
                
            else:
                for j, roughness in enumerate(roughnesses):
                    roughness_label = re.sub(r'[\W_]', ' ', roughness)
                    
                    fig = plt.figure()
                    auxa = polar_demo.fractional_polar_axes(
                            fig, thlim=(0,90), rlim=(0,r_max), step=(15,r_step),
                            thlabel='Elevation angle',
                            rlabel='Reflectance factor')
                    
                    # Full scattering phase functions
                    particle_radii = list()
                    theta_deg = list()
                    brfs = list()
                    for RE, file_path in self.data_I[shape][roughness].items():
                        particle_radii.append(np.around(float(RE)))
                        (brf,
                         theta_rad,
                         mean_wvls) = self.bi_directional_reflectance_factor(
                                                                     file_path, 
                                                                     n_bins)
                        theta_deg.append(np.rad2deg(theta_rad))
                        brfs.append(brf)
                
                    idxs = np.argsort(particle_radii)
                
                    for k, idx in enumerate(idxs):
                        auxa.plot(theta_deg[idx][::-1], brfs[idx],
                                  color=colors[k])
                                  
                    # Henyey Greenstein scattering phase functions
                    particle_radii = list()
                    theta_deg = list()
                    brfs = list()
                    for RE, file_path in self.data_HG[shape][roughness].items():
                        particle_radii.append(np.around(float(RE)))
                        (brf,
                         theta_rad,
                         mean_wvls) = self.bi_directional_reflectance_factor(
                                                                     file_path, 
                                                                     n_bins)
                        theta_deg.append(np.rad2deg(theta_rad))
                        brfs.append(brf)
                
                    idxs = np.argsort(particle_radii)
                
                    for k, idx in enumerate(idxs):
                        auxa.plot(theta_deg[idx][::-1], brfs[idx],
                                  label='%d' % particle_radii[idx], 
                                  color=colors[k], linestyle='dashed')
                              
                    #plt.title('%dnm %d$^{\circ}$ BRFs of ice %s %ss' % (
                     #                                          wvl_nm,
                      #                                         zenith,
                       #                                        roughness_label,
                        #                                       shape_label))
                    plt.title('%s %ss' % (roughness_label, shape_label))
                    plt.legend(bbox_to_anchor=(1, 1),
                               bbox_transform=plt.gcf().transFigure,
                               title='RE ($\mathrm{\mu m}$)',
                               fontsize='small')
                               
                    if savefigs:
                        figname = '%s_%s_%dbrf.pdf' % (roughness, shape, zenith)
                        plt.savefig(figname)
                    else:
                        plt.show()
                        plt.close()
    
    def plot_bidirectional_reflectance_factor(self, theta_r, active_area=1.,
                                              d_dome=175., markersize=8,
                                              xlim=(10,1010),
                                              ylim=(0,1),
                                              savefig=False,
                                              legend_font=10):
        """ Plot bi-directional theta_r (deg.) reflectance factor as a function
            of particle effective radius with:
            
            active_area = active area of photodiode (mm)
            d_dome = diameter of hemisphere    
        """
        n_bins = calculate_bins(active_area, d_dome)
        theta_r_rad = np.deg2rad(theta_r)
        
        fig = plt.figure()
        wvl_nm = np.around(float(self.args['wvl']) * 10**3)
        zenith = np.around(float(self.args['theta_0']))
        theta_r_display = np.around(theta_r)
        
        num_shapes = len(self.args['shapes'])
        num_colors = num_shapes
        color_idxs = np.arange(num_colors)
        for i, shape in enumerate(self.args['shapes']):
            if shape == 'sphere':
                num_colors = num_colors - 1
        
        color_list = plt.cm.Dark2(np.linspace(0, 1, num_colors))
        for i, shape in enumerate(self.args['shapes']):
            shape_label = re.sub(r'[\W_]', ' ', shape)
            print('Calculating and plotting reflectance factors for %ss...' % 
                                                                   shape_label)
            if shape == 'sphere':
                color_idxs = color_idxs - 1
                
                # Full scattering phase functions
                particle_radii = list()
                brf = list()
                for RE, file_path in self.data_I[shape].items():
                    particle_radii.append(float(RE))
                    (brf_all,
                     theta_rad,
                     mean_wvls) = self.bi_directional_reflectance_factor(
                                                                     file_path, 
                                                                     n_bins)
                    # Find nearest theta_r
                    idx = (np.absolute(theta_rad - theta_r_rad)).argmin()
                    brf.append(brf_all[idx])
                
                particle_radii = np.array(particle_radii)
                brf = np.array(brf)
                
                idxs = np.argsort(particle_radii)
                
                plt.plot(particle_radii[idxs], brf[idxs], color='k',
                         marker='o', markersize=markersize)
                
                # Henyey Greenstein scattering phase functions
                particle_radii = list()
                brf = list()
                for RE, file_path in self.data_HG[shape].items():
                    particle_radii.append(float(RE))
                    (brf_all,
                     theta_rad,
                     mean_wvls) = self.bi_directional_reflectance_factor(
                                                                     file_path, 
                                                                     n_bins)
                    # Find nearest theta_r
                    idx = (np.absolute(theta_rad - theta_r_rad)).argmin()
                    brf.append(brf_all[idx])
                
                particle_radii = np.array(particle_radii)
                brf = np.array(brf)
                
                idxs = np.argsort(particle_radii)                
                
                plt.plot(particle_radii[idxs], brf[idxs], color='k',
                         marker='o', linestyle='dashed',
                         label='%ss' % shape_label,
                         markersize=markersize)
            else:
                color = color_list[color_idxs[i]]
                for roughness, RE_data_I in self.data_I[shape].items():
                    roughness_label = re.sub(r'[\W_]', ' ', roughness)                    
                    """ Full scattering phase function
                    """
                    if roughness == 'smooth':
                        marker = 'o'
                    elif roughness == 'moderately_rough':
                        marker = 'd'
                    elif roughness == 'severely_rough':
                        marker = '*'
                    
                    particle_radii = list()
                    brf = list()
                    for RE, file_path in RE_data_I.items():
                        particle_radii.append(float(RE))
                        (brf_all,
                         theta_rad,
                         mean_wvls) = self.bi_directional_reflectance_factor(
                                                                    file_path, 
                                                                    n_bins)
                        # Find nearest theta_r
                        idx = (np.absolute(theta_rad - theta_r_rad)).argmin()
                        brf.append(brf_all[idx])
                
                    particle_radii = np.array(particle_radii)
                    brf = np.array(brf)
                
                    idxs = np.argsort(particle_radii)                    
                    
                    plt.plot(particle_radii[idxs], brf[idxs],
                             color=color, marker=marker, markersize=markersize,
                             label='%s %ss' % (roughness_label, shape_label))
                
                for roughness, RE_data_HG in self.data_HG[shape].items():
                    """ Henyey Greenstein scattering phase function
                    """
                    if roughness == 'smooth':
                        marker = 'o'
                    elif roughness == 'moderately_rough':
                        marker = 'd'
                    elif roughness == 'severely_rough':
                        marker = '*'
                    
                    particle_radii = list()
                    brf = list()
                    for RE, file_path in RE_data_HG.items():
                        particle_radii.append(float(RE))
                        (brf_all,
                         theta_rad,
                         mean_wvls) = self.bi_directional_reflectance_factor(
                                                                    file_path, 
                                                                    n_bins)
                        # Find nearest theta_r
                        idx = (np.absolute(theta_rad - theta_r_rad)).argmin()
                        brf.append(brf_all[idx])
                
                    particle_radii = np.array(particle_radii)
                    brf = np.array(brf)
                
                    idxs = np.argsort(particle_radii)                    
                    
                    plt.plot(particle_radii[idxs], brf[idxs],
                             color=color, marker=marker,
                             linestyle='dashed', markersize=markersize)
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        plt.xlabel('Ice particle effective radius ($\mathrm{\mu m}$)')
        plt.ylabel('Reflectance factor')
        plt.title('%dnm %d$^{\circ}$;%d$^{\circ}$ bi-directional reflectance '
                  'factors' % (wvl_nm, zenith, theta_r_display))
        plt.legend(loc=1, fontsize=legend_font)
        plt.grid()
        
        if savefig:
            figname = '%dnm_%d-%dbrfs.pdf' % (wvl_nm, zenith, theta_r_display)
            plt.savefig(figname)
        else:
            plt.show()
            plt.close()
        
    def plot_directional_hemispherical_reflectance(self, markersize=8,
                                                   xlim=(10,1010),
                                                   ylim=(0,1),
                                                   savefig=False,
                                                   legend_font=10):
        """ Plot directional-hemispherical reflectance as a function of
            particle effective radius
        """
        fig = plt.figure()
        wvl_nm = np.around(float(self.args['wvl']) * 10**3)
        zenith = np.around(float(self.args['theta_0']))
        
        num_shapes = len(self.args['shapes'])
        num_colors = num_shapes
        color_idxs = np.arange(num_colors)
        for i, shape in enumerate(self.args['shapes']):
            if shape == 'sphere':
                num_colors = num_colors - 1
        
        color_list = plt.cm.Dark2(np.linspace(0, 1, num_colors))
        for i, shape in enumerate(self.args['shapes']):
            shape_label = re.sub(r'[\W_]', ' ', shape)
            print('Calculating and plotting albedo for %ss...' % shape_label)         
            
            if shape == 'sphere':
                color_idxs = color_idxs - 1
                
                # Full scattering phase functions
                particle_radii = list()
                albedo = list()
                for RE, file_path in self.data_I[shape].items():
                    particle_radii.append(float(RE))
                    albedo.append(self.directional_hemispherical_reflectance(
                                                                  file_path)[0])
                
                particle_radii = np.array(particle_radii)
                albedo = np.array(albedo)
                
                idxs = np.argsort(particle_radii)
                
                plt.plot(particle_radii[idxs], albedo[idxs], color='k',
                         marker='o', markersize=markersize)
                
                # Henyey Greenstein scattering phase functions
                particle_radii = list()
                albedo = list()
                for RE, file_path in self.data_HG[shape].items():
                    particle_radii.append(float(RE))
                    albedo.append(self.directional_hemispherical_reflectance(
                                                                  file_path)[0])
                
                particle_radii = np.array(particle_radii)
                albedo = np.array(albedo)
                
                idxs = np.argsort(particle_radii)                
                
                plt.plot(particle_radii[idxs], albedo[idxs], color='k',
                         marker='o', linestyle='dashed',
                         label='%ss' % shape_label, 
                         markersize=markersize)
            else:
                color = color_list[color_idxs[i]]
                for roughness, RE_data_I in self.data_I[shape].items():
                    roughness_label = re.sub(r'[\W_]', ' ', roughness)
                    
                    """ Full scattering phase function
                    """
                    if roughness == 'smooth':
                        marker = 'o'
                    elif roughness == 'moderately_rough':
                        marker = 'd'
                    elif roughness == 'severely_rough':
                        marker = '*'
                    
                    particle_radii = list()
                    albedo = list()
                    for RE, file_path in RE_data_I.items():
                        particle_radii.append(float(RE))
                        albedo.append(
                                     self.directional_hemispherical_reflectance(
                                                                  file_path)[0])
                
                    particle_radii = np.array(particle_radii)
                    albedo = np.array(albedo)
                
                    idxs = np.argsort(particle_radii)                    
                    
                    plt.plot(particle_radii[idxs], albedo[idxs],
                             color=color, marker=marker, markersize=markersize,
                             label='%s %ss' % (roughness_label, shape_label))
                
                for roughness, RE_data_HG in self.data_HG[shape].items():
                    """ Henyey Greenstein scattering phase function
                    """
                    if roughness == 'smooth':
                        marker = 'o'
                    elif roughness == 'moderately_rough':
                        marker = 'd'
                    elif roughness == 'severely_rough':
                        marker = '*'
                    
                    particle_radii = list()
                    albedo = list()
                    for RE, file_path in RE_data_HG.items():
                        particle_radii.append(float(RE))
                        albedo.append(
                                     self.directional_hemispherical_reflectance(
                                                                  file_path)[0])
                
                    particle_radii = np.array(particle_radii)
                    albedo = np.array(albedo)
                
                    idxs = np.argsort(particle_radii)                    
                    
                    plt.plot(particle_radii[idxs], albedo[idxs],
                             color=color, marker=marker, linestyle='dashed',
                             markersize=markersize)
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        plt.xlabel('Ice particle effective radius ($\mathrm{\mu m}$)')
        plt.ylabel('Reflectance')
        plt.title('%dnm %d$^{\circ}$ directional-hemispherical reflectance'
                  % (wvl_nm, zenith))
        plt.legend(loc=1, fontsize=legend_font)
        plt.grid()
        
        if savefig:
            figname = '%dnm_%dalbedo.pdf' % (wvl_nm, zenith)
            plt.savefig(figname)
        else:
            plt.show()
            plt.close()
    
    def bi_directional_reflectance_factor(self, file_path, n_bins):
        """ Read in data and calculate bi-directional reflectance factors
        """
        data_file = pd.read_csv(file_path, delim_whitespace=True)
        mean_wvls = 1. / data_file['wvn[um^-1]'].mean()
        wvl0 = float(self.args['wvl'])
        
        if np.absolute(wvl0 - mean_wvls) > 0.1:
            brf = None
            midpoints = None
        else:
            hist_range = (0., np.pi/2)
            
            
            Q_down = data_file['wvn[um^-1]'].sum()
            weights = data_file[data_file.condition==1]['wvn[um^-1]']
            theta_exit = data_file[data_file.condition==1]['theta_n']
            
            h = np.histogram(theta_exit, bins=n_bins, range=hist_range,
                             weights=weights)
            
            midpoints = (np.diff(h[1]) / 2.) + h[1][:-1]
            brf_weights = (np.sin(midpoints) * np.cos(midpoints) / 
                           np.sum(np.sin(midpoints) * np.cos(midpoints)))
            
            brf = h[0] / (Q_down * brf_weights)
            
        return(brf, midpoints, mean_wvls)
    
    def directional_hemispherical_reflectance(self, file_path):
        """ Read in data and calculate directional-hemispherical 
            reflectance, a.k.a. black sky albedo
        """
        data_file = pd.read_csv(file_path, delim_whitespace=True)
        mean_wvls = 1. / data_file['wvn[um^-1]'].mean()
        wvl0 = float(self.args['wvl'])
        
        if np.absolute(wvl0 - mean_wvls) > 0.1:
            albedo = None
        else:
            Q_down = data_file['wvn[um^-1]'].sum()
            Q_up = data_file[data_file.condition==1]['wvn[um^-1]'].sum()
        
            albedo = Q_up / Q_down
        
        return(albedo, mean_wvls)
        
    def mean_wvl(self, file_path):
        """ Calculate mean wavelength in output
        """
        data_file = pd.read_csv(file_path, delim_whitespace=True)
        
def plot_spectral_albedo(top_data_dir='/data3/amaschne/AGU2015_60zenith',
                         shape='sphere',
                         roughness='smooth',
                         wvls=np.arange(0.305, 3.005, 0.01),
                         half_width=1e-12,
                         n_photon=1000000,
                         theta_0=60.0,
                         Stokes_0=[1,0,0,0] ,
                         Henyey_Greenstein=True,
                         xmin=0.2,
                         xmax=3.0,
                         savefig=False):
    """ Plot spectral directional-hemispherical reflectance for a given
        shape habit and roughness.
    """
    
    rds_snw_list = list()
    albedo = dict()
    valid_wvls = dict()
    for i, wvl in enumerate(wvls):
        data_set = MonteCarloDataSet(top_data_dir=top_data_dir, shapes=[shape],
                                     roughnesses=[roughness], wvl=wvl, 
                                     half_width=half_width, n_photon=n_photon, 
                                     theta_0=theta_0, Stokes_0=Stokes_0)
        if Henyey_Greenstein:
            if shape=='sphere':
                data = data_set.data_HG[shape]
            else:
                data = data_set.data_HG[shape][roughness]
        else:
            if shape=='sphere':
                data = data_set.data_I[shape]
            else:
                data = data_set.data_I[shape][roughness]
        
        if len(data.keys()) > 0:
            for RE, file_path in data.items():
                 if not RE in rds_snw_list:
                     rds_snw_list.append(RE)
                     valid_wvls[RE] = list()
                     albedo[RE] = list()
                     
                 rho, wvl = data_set.directional_hemispherical_reflectance(
                                                                      file_path)
                 valid_wvls[RE].append(wvl)
                 albedo[RE].append(rho)
                 
                 wvl_nm = np.around(wvl*1000)
                 RE_um = np.around(float(RE))
                 rho = np.around(rho, decimals=3)
                 print('rho(wvl=%dnm, RE=%dum) = %s' % (wvl_nm, RE_um, rho))
    
    fig = plt.figure()
    
    zenith = np.around(theta_0)
    radii = natsort.natsorted(rds_snw_list)
    for i, RE in enumerate(radii):
        label = np.around(float(RE))
        plt.plot(valid_wvls[RE], albedo[RE], label='%d' % label)
    
    plt.legend(title='Particle effective radius ($\mathrm{\mu m}$)', loc=1)
    plt.grid()
    plt.xlim((0.2, 3.0))
    plt.xticks(np.arange(xmin, xmax + 0.1, 0.2))
    plt.ylim((0.0,1.0))
    plt.yticks(np.arange(0, 1.1, 0.1))
    
    plt.xlabel('Wavelength ($\mathrm{\mu m}$)')
    plt.ylabel('Reflectance')
    plt.title('%d$^{\circ}$ directional-hemispherical reflectance for '
              'ice %ss'% (zenith, shape))
              
    if savefig:
        figname = 'spectral_%dalbedo_%s.pdf' % (zenith, shape)
        plt.savefig(figname)
    else:
        plt.show()
        plt.close()
    
def calculate_bins(active_area, d_dome):
    """ Calculate number of bins to simulate photodiode with given active area
        mounted in dome with given diameter

        Returns number of bins n_bins
    """
    n_bins =  int((np.pi * d_dome) / (4.*active_area))
    print('BRF computing over %d bins' % n_bins)
    
    return n_bins

def main():
    data = MonteCarloDataSet()
    data.plot_directional_hemispherical_reflectance()

if __name__=='__main__':
    main()