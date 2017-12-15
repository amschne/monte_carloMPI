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
from matplotlib.projections.geo import HammerAxes
from matplotlib.projections.geo import AitoffAxes
import matplotlib.projections as mprojections
from matplotlib.axes import Axes
from matplotlib.patches import Wedge
import matplotlib.spines as mspines

import ipdb

import polar_demo

DENSITY_ICE = 917.0 # kg / m^3

def get_args():
    """ User input
    """
    #top_data_dir = '/data1/amaschne/nerd_all'
    top_data_dir = '/data1/amaschne/nerd_ssa_cal' # for comparing SSA to NERD BRFs
    shapes = ['sphere',
              'droxtal',
              'solid_hexagonal_column', 
              #'hollow_hexagonal_column',
              #'8-element_column_aggregate',
              #'hexagonal_plate',
              #'5-element_plate_aggregate',
              #'10-element_plate_aggregate',
              #'solid_bullet_rosette',
              #'hollow_bullet_rosette',
             ]
    roughnesses = ['smooth'#, 'moderately_rough','severely_rough'
                   ]
    wvl = 1.3 # um
    half_width = 0.085 # um
    n_photon = 250000
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
        
        self.obs_dict = dict()
    
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

    def add_1550_data(self):    
        self.data_I_1550 = dict()
        self.data_HG_1550 = dict()
        for i, shape in enumerate(self.args['shapes']): 
            self.data_I_1550[shape] = dict()
            self.data_HG_1550[shape] = dict()
            
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
                    files_I, files_HG = self.get_1550_files(data_dir)
                
                    for k, file_tuple in enumerate(files_I):
                        file_name = file_tuple[0]
                        RE = file_tuple[1]
                        self.data_I_1550[shape][RE] = os.path.join(data_dir,
                                                                   file_name)
                
                    for k, file_tuple in enumerate(files_HG):
                        file_name = file_tuple[0]
                        RE = file_tuple[1]
                        self.data_HG_1550[shape][RE] = os.path.join(data_dir,
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
                        self.data_I_1550[shape][roughness] = dict()
                        self.data_HG_1550[shape][roughness] = dict()
                    
                        files_I, files_HG = self.get_1550_files(data_dir)
                    
                        for k, file_tuple in enumerate(files_I):
                            file_name = file_tuple[0]
                            RE = file_tuple[1]
                            self.data_I_1550[shape][roughness][RE] = os.path.join(
                                                                      data_dir,
                                                                      file_name)
                    
                        for k, file_tuple in enumerate(files_HG):
                            file_name = file_tuple[0]
                            RE = file_tuple[1]
                            self.data_HG_1550[shape][roughness][RE] = os.path.join(
                                                                      data_dir,
                                                                      file_name)    
    
    def get_1550_files(self, data_dir):
        """
        """
        expression_I = '%s_%s_*_%s_%s_I%s%s%s%s.txt' % ('1.55',
                                                       '0.13',
                                                       self.args['n_photon'],
                                                       self.args['theta_0'],
                                                       self.args['Stokes_0'][0],
                                                       self.args['Stokes_0'][1],
                                                       self.args['Stokes_0'][2],
                                                       self.args['Stokes_0'][3])
        expression_HG = '%s_%s_*_%s_%s_HG.txt' % ('1.55',
                                                  '0.13',
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
    
    def plot_brf_3D(self,
                    shapes=list(),
                    roughnesses=list(),
                    active_area=1.,
                    d_dome=175.,
                    nlevels=100,
                    rmax=1,
                    savefigs=False,
                    theta_bins=9,
                    phi_bins=4):
        """ Plot azimuthal BRFs for different grain sizes for a list of given
            shapes and roughnesses.
        """
        mprojections.register_projection(UpperHammerAxes)
        mprojections.register_projection(UpperAitoffAxes)
        
        wvl_nm = np.around(float(self.args['wvl']) * 10**3)
        zenith = np.around(float(self.args['theta_0']))
        
        phi_range = (0., 2*np.pi)
        theta_range = (0., np.pi/2)
        
        levels = np.linspace(0, rmax, nlevels)
        ticks = np.arange(0, 1.1, 0.1) * rmax
        cmap = plt.cm.get_cmap("gray")
        
        if theta_bins==None:
            theta_bins = calculate_bins(active_area, d_dome)
        if phi_bins==None:
            phi_bins = theta_bins*4    
        
        if len(shapes)==0:
            shapes=self.args['shapes']
        if len(roughnesses)==0:
            roughnesses=self.args['roughnesses']
        
        for i, shape in enumerate(shapes):
            shape_label = re.sub(r'[\W_]', ' ', shape)
            
            if shape=='sphere':
                # Full scattering phase functions
                particle_radii = list()
                phi_rad = list()
                theta_rad = list()
                theta_deg = list()
                brfs = list()
                for RE, file_path in self.data_I[shape].items():
                    particle_radii.append(np.around(float(RE)))
                    (brf,
                     midpoints,
                     mean_wvls) = self.bi_directional_reflectance_factor_3D(
                                                                     file_path, 
                                                                     phi_bins,
                                                                     theta_bins)
                    
                    phi_rad.append(midpoints[0])
                    theta_rad.append(midpoints[1])
                    theta_deg.append(np.rad2deg(midpoints[1]))
                    brfs.append(brf)
                
                idxs = np.argsort(particle_radii)
                
                for k, idx in enumerate(idxs):
                    fig, ax = plt.subplots(subplot_kw=dict(
                                                projection='upper_hammer'))
                    cax = ax.contourf(phi_rad[idx] - np.pi, 
                                      theta_rad[idx][::-1],
                                      brfs[idx],
                                      levels,
                                      cmap=cmap)
                    
                    ax.grid(True)
                    cb = fig.colorbar(cax, ticks=ticks, 
                                      orientation='horizontal')
                    cb.set_label("Reflectance factor")
                    plt.title('%d $\mathrm{\mu m}$ %ss' % (particle_radii[idx], 
                                                           shape_label))
                    plt.show()

                # Henyey Greenstein scattering phase functions
                particle_radii = list()
                phi_rad = list()
                theta_rad = list()
                theta_deg = list()
                brfs = list()
                for RE, file_path in self.data_HG[shape].items():
                    particle_radii.append(np.around(float(RE)))
                    (brf,
                     midpoints,
                     mean_wvls) = self.bi_directional_reflectance_factor_3D(
                                                                     file_path, 
                                                                     phi_bins,
                                                                     theta_bins)
                                                                     
                    phi_rad.append(midpoints[0])
                    theta_rad.append(midpoints[1])
                    theta_deg.append(np.rad2deg(midpoints[1]))
                    brfs.append(brf)
                
                idxs = np.argsort(particle_radii)
                
                for k, idx in enumerate(idxs):
                    fig, ax = plt.subplots(subplot_kw=dict(
                                                projection='upper_aitoff'))
                    cax = ax.contourf(phi_rad[idx] - np.pi, 
                                      theta_rad[idx][::-1],
                                      brfs[idx],
                                      levels,
                                      cmap=cmap)
                    
                    ax.grid(True)
                    cb = fig.colorbar(cax, ticks=ticks, orientation='horizontal')
                    cb.set_label("Reflectance factor")
                    #plt.title('%d $\mathrm{\mu m}$ %ss (HG)'
                    #                                   % (particle_radii[idx], 
                    #                                      shape_label))
                    plt.show()
                    
            else:
                for j, roughness in enumerate(roughnesses):
                    roughness_label = re.sub(r'[\W_]', ' ', roughness)
                    
                    # Full scattering phase functions
                    particle_radii = list()
                    phi_rad = list()
                    theta_rad = list()
                    theta_deg = list()
                    brfs = list()
                    for RE, file_path in self.data_I[shape][roughness].items():
                        particle_radii.append(np.around(float(RE)))
                        (brf,
                         midpoints,
                         mean_wvls) = self.bi_directional_reflectance_factor_3D(
                                                                     file_path, 
                                                                     phi_bins,
                                                                     theta_bins)
                        phi_rad.append(midpoints[0])
                        theta_rad.append(midpoints[1])
                        theta_deg.append(np.rad2deg(midpoints[1]))
                        brfs.append(brf)
                
                    idxs = np.argsort(particle_radii)
                
                    for k, idx in enumerate(idxs):
                        fig, ax = plt.subplots(subplot_kw=dict(
                                                projection='upper_hammer'))
                        cax = ax.contourf(phi_rad[idx] - np.pi, 
                                          theta_rad[idx][::-1],
                                          brfs[idx],
                                          levels,
                                          cmap=cmap)
                        ax.grid(True)
                        cb = fig.colorbar(cax, ticks=ticks, 
                                          orientation='horizontal')
                        cb.set_label("Reflectance factor")
                        plt.title('%d $\mathrm{\mu m}$ %s %ss'
                                                        % (particle_radii[idx], 
                                                           roughness_label,
                                                           shape_label))
                        plt.show()
                        
                    # Henyey Greenstein scattering phase functions
                    particle_radii = list()
                    phi_rad = list()
                    theta_rad = list()
                    theta_deg = list()
                    brfs = list()
                    for RE, file_path in self.data_HG[shape][roughness].items():
                        particle_radii.append(np.around(float(RE)))
                        (brf,
                         midpoints,
                         mean_wvls) = self.bi_directional_reflectance_factor_3D(
                                                                     file_path, 
                                                                     phi_bins,
                                                                     theta_bins)
                        phi_rad.append(midpoints[0])
                        theta_rad.append(midpoints[1])
                        theta_deg.append(np.rad2deg(midpoints[1]))
                        brfs.append(brf)
                
                    idxs = np.argsort(particle_radii)
                
                    for k, idx in enumerate(idxs):
                        fig, ax = plt.subplots(subplot_kw=dict(
                                                projection='upper_hammer'))
                        cax = ax.contourf(phi_rad[idx] - np.pi, 
                                          theta_rad[idx][::-1],
                                          brfs[idx],
                                          levels,
                                          cmap=cmap)
                    
                        ax.grid(True)
                        cb = fig.colorbar(cax, ticks=ticks, 
                                          orientation='horizontal')
                        cb.set_label("Reflectance factor")
                        plt.title('%d $\mathrm{\mu m}$ %s %ss (HG)'
                                                        % (particle_radii[idx], 
                                                           roughness_label,
                                                           shape_label))
                        plt.show()
                            
    def plot_brf_all_angles(self,
                            shapes=list(),
                            roughnesses=list(),
                            theta_bins=18,
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
        
        if theta_bins == None:
            n_bins = calculate_bins(active_area, d_dome)
        else:
            n_bins = theta_bins
        
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
                        
    def plot_1300_1550_diff(self, theta_r, active_area=1.,
                            d_dome=175., markersize=8,
                            xlim=(10,1010),
                            ylim=(0,1),
                            savefig=False,
                            legend_font=10,
                            theta_bins=None):
        """
        """
        self.add_1550_data()
        
        if theta_bins == None:
            n_bins = calculate_bins(active_area, d_dome)
        else:
            n_bins = theta_bins
        
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
                brf_1550 = list()
                for RE, file_path in self.data_I[shape].items():
                    file_path_1550 = self.data_I_1550[shape][RE]
                    particle_radii.append(float(RE))
                    (brf_all,
                     theta_rad,
                     mean_wvls) = self.bi_directional_reflectance_factor(
                                                                     file_path, 
                                                                     n_bins)
                                                                     
                    (brf_all_1550,
                     theta_rad_1550,
                     mean_wvls_1550) = self.bi_directional_reflectance_factor(
                                                                file_path_1550,
                                                                n_bins,
                                                                wvl_check=False)
                    # Find nearest theta_r
                    idx = (np.absolute(theta_rad - theta_r_rad)).argmin()
                    brf.append(brf_all[idx])
                    brf_1550.append(brf_all_1550[idx])
                
                particle_radii = np.array(particle_radii)
                brf = np.array(brf)
                brf_1550 = np.array(brf_1550)
                brf_diff = brf - brf_1550
                
                idxs = np.argsort(particle_radii)
                
                plt.plot(particle_radii[idxs], brf_diff[idxs], color='k',
                         marker='o', markersize=markersize)
                
                # Henyey Greenstein scattering phase functions
                particle_radii = list()
                brf = list()
                brf_1550 = list()
                for RE, file_path in self.data_HG[shape].items():
                    file_path_1550 = self.data_HG_1550[shape][RE]
                    particle_radii.append(float(RE))
                    (brf_all,
                     theta_rad,
                     mean_wvls) = self.bi_directional_reflectance_factor(
                                                                     file_path, 
                                                                     n_bins)
                                                                     
                    (brf_all_1550,
                     theta_rad_1550,
                     mean_wvls_1550) = self.bi_directional_reflectance_factor(
                                                                file_path_1550,
                                                                n_bins,
                                                                wvl_check=False)
                    # Find nearest theta_r
                    idx = (np.absolute(theta_rad - theta_r_rad)).argmin()
                    brf.append(brf_all[idx])
                    brf_1550.append(brf_all_1550[idx])
                
                particle_radii = np.array(particle_radii)
                brf = np.array(brf)
                brf_1550 = np.array(brf_1550)
                brf_diff = brf - brf_1550
                
                idxs = np.argsort(particle_radii)                
                
                plt.plot(particle_radii[idxs], brf_diff[idxs], color='k',
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
                    brf_1550 = list()
                    for RE, file_path in RE_data_I.items():
                        file_path_1550 = self.data_I_1550[shape][roughness][RE]
                        particle_radii.append(float(RE))
                        (brf_all,
                         theta_rad,
                         mean_wvls) = self.bi_directional_reflectance_factor(
                                                                    file_path, 
                                                                    n_bins)
                                                                    
                        (brf_all_1550,
                         theta_rad_1550,
                         mean_wvls_1550)=self.bi_directional_reflectance_factor(
                                                            file_path_1550,
                                                            n_bins,
                                                            wvl_check=False)
                        # Find nearest theta_r
                        idx = (np.absolute(theta_rad - theta_r_rad)).argmin()
                        brf.append(brf_all[idx])
                        brf_1550.append(brf_all_1550[idx])
                
                    particle_radii = np.array(particle_radii)
                    brf = np.array(brf)
                    brf_1550 = np.array(brf_1550)
                    brf_diff = brf - brf_1550
                
                    idxs = np.argsort(particle_radii)                    
                    
                    plt.plot(particle_radii[idxs], brf_diff[idxs],
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
                    brf_1550 = list()
                    for RE, file_path in RE_data_HG.items():
                        file_path_1550 = self.data_HG_1550[shape][roughness][RE]
                        particle_radii.append(float(RE))
                        (brf_all,
                         theta_rad,
                         mean_wvls) = self.bi_directional_reflectance_factor(
                                                                    file_path, 
                                                                    n_bins)
                                                                    
                        (brf_all_1550,
                         theta_rad_1550,
                         mean_wvls_1550)=self.bi_directional_reflectance_factor(
                                                                file_path_1550,
                                                                n_bins,
                                                                wvl_check=False)
                        # Find nearest theta_r
                        idx = (np.absolute(theta_rad - theta_r_rad)).argmin()
                        brf.append(brf_all[idx])
                        brf_1550.append(brf_all_1550[idx])
                
                    particle_radii = np.array(particle_radii)
                    brf = np.array(brf)
                    brf_1550 = np.array(brf_1550)
                    brf_diff = brf - brf_1550
                
                    idxs = np.argsort(particle_radii)
                    
                    plt.plot(particle_radii[idxs], brf_diff[idxs],
                             color=color, marker=marker,
                             linestyle='dashed', markersize=markersize)
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        plt.xlabel('Ice particle effective radius ($\mathrm{\mu m}$)')
        plt.ylabel('$R_{1300 \mathrm{nm}} - R_{1550 \mathrm{nm}}$')
        plt.title('Difference in %dnm and %dnm %d$^{\circ}$;%d$^{\circ}$ '
                  'bi-directional reflectance '
                  'factors' % (wvl_nm, 1550, zenith, theta_r_display))
        plt.legend(loc=1, fontsize=legend_font)
        plt.grid()
        
        if savefig:
            figname = 'Diff%dnm%dnm_%d-%dbrfs.pdf' % (wvl_nm, 1550, zenith,
                                                      theta_r_display)
            plt.savefig(figname)
        else:
            plt.show()
            plt.close()
    
    def add_observational_data(self, ssa, brfs, label):
        #r_eff_mean = np.mean(r_eff)
        #r_eff_std = np.std(r_eff)
        mean_brf = np.mean(brfs)
        std_brf = np.std(brfs)

        self.obs_dict[label] = {'ssa' : ssa,
                                'brf' : mean_brf,
                                'brf_std' : std_brf}
        
    def overlay_nerd_obs(self):
        num_of_obs = len(self.obs_dict.keys())
        color_list = plt.cm.rainbow(np.linspace(0, 1, num_of_obs))
        
        color_i = 0
        for label, obs_dict in sorted(self.obs_dict.items()):
            print label
            plt.errorbar(obs_dict['brf'], obs_dict['ssa'], 
                         xerr=obs_dict['brf_std'], color=color_list[color_i],
                         marker='o', label=label)
            color_i+=1
    def overlay_nerd_obs_old(self):
        marker_list = ['^', 's', '*', '+', 'x']
        i = 0
        for name, vals in self.obs_dict.items():
            for j, brf in enumerate(vals['']):
                if j==0:
                    plt.errorbar(vals['RE'], brf, xerr=vals['xerr'],
                                 fmt=marker_list[i], color='b', label=name)
                else:
                    plt.errorbar(vals['RE'], brf, xerr=vals['xerr'],
                                 fmt=marker_list[i], color='b')                    
                             
            i += 1
    
    def plot_bidirectional_reflectance_factor(self, theta_r, active_area=1.,
                                              d_dome=175., markersize=8,
                                              xlim=(0,0.7),
                                              ylim=(0,70),
                                              savefig=False,
                                              legend_font=10,
                                              theta_bins=18,
                                              overlay_nerd_obs=False):
        """ Plot bi-directional theta_r (deg.) reflectance factor as a function
            of particle effective radius with:
            
            active_area = active area of photodiode (mm)
            d_dome = diameter of hemisphere    
        """
        if theta_bins == None:
            n_bins = calculate_bins(active_area, d_dome)
        else:
            n_bins = theta_bins
        
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
                
                plt.plot(brf[idxs], (3.*10**6)/(particle_radii[idxs] *
                                                DENSITY_ICE),
                         color='k',
                         marker=None, linestyle='dashed',
                         label='%ss' % shape_label)
            else:
                color = color_list[color_idxs[i]]
                for roughness, RE_data_I in self.data_I[shape].items():
                    roughness_label = re.sub(r'[\W_]', ' ', roughness)                    
                    """ Full scattering phase function
                    """
                    if roughness == 'smooth':
                        marker = None
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
                    
                    plt.plot(brf[idxs], (3.*10**6) / (particle_radii[idxs] * 
                                                      DENSITY_ICE),
                             color=color, marker=marker,
                             label='%ss' % (#roughness_label,
                                                          shape_label))
                
                for roughness, RE_data_HG in self.data_HG[shape].items():
                    """ Henyey Greenstein scattering phase function
                    """
                    if roughness == 'smooth':
                        marker = None
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
                    
                    plt.plot(brf[idxs], (3.*10**6) / (particle_radii[idxs] *
                                                      DENSITY_ICE),
                             color=color, marker=marker,
                             linestyle='dashed')
        
        plt.xlim(xlim)
        plt.ylim(ylim)
        
        if overlay_nerd_obs:
            self.overlay_nerd_obs()
        
        plt.ylabel('Specific surface area ($\mathrm{m^2/kg}$)')
        plt.xlabel('Reflectance factor')
        plt.title('Snow SSA vs. %dnm %d$^{\circ}$;%d$^{\circ}$ BRFs'
                  % (wvl_nm, zenith, theta_r_display))
        plt.legend()#loc=1, fontsize=legend_font)
        plt.grid()
        
        if savefig:
            figname = '%dnm_%d-%dbrfs_vs_ssa.pdf' % (wvl_nm, zenith, 
                                                     theta_r_display)
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
    
    def bi_directional_reflectance_factor_3D(self, file_path, phi_bins, 
                                             theta_bins):
        """ Read in data and calculate 3D bi-directional reflectance factors
        """
        data_file = pd.read_csv(file_path, delim_whitespace=True)
        mean_wvls = 1. / data_file['wvn[um^-1]'].mean()
        wvl0 = float(self.args['wvl'])
        
        if np.absolute(wvl0 - mean_wvls) > 0.1:
            brf = None
            midpoints = None
        else:
            phi_range = (0., 2*np.pi)
            theta_range = (0., np.pi/2)
            
            Q_down = data_file['wvn[um^-1]'].sum()
            weights = data_file[data_file.condition==1]['wvn[um^-1]']
            phi_exit = data_file[data_file.condition==1]['phi_n']
            theta_exit = data_file[data_file.condition==1]['theta_n']
            
            h = np.histogram2d(phi_exit,
                               theta_exit,
                               bins=[phi_bins, theta_bins],
                               range=[phi_range, theta_range],
                               weights=weights)
            phi_midpoints = (np.diff(h[1]) / 2.) + h[1][:-1]
            theta_midpoints = (np.diff(h[2]) / 2.) + h[2][:-1]
            midpoints = [phi_midpoints, theta_midpoints]
            
            theta_weights = (np.sin(theta_midpoints) * np.cos(theta_midpoints)/ 
                     np.sum(np.sin(theta_midpoints) * np.cos(theta_midpoints)))
                     
            brf_weights = theta_weights / phi_bins
                              
            brf = (h[0] / (Q_down * brf_weights)).T
            #print brf
        return(brf, midpoints, mean_wvls)
    
    def bi_directional_reflectance_factor(self, file_path, n_bins,
                                          wvl_check=True):
        """ Read in data and calculate bi-directional reflectance factors
        """
        #print file_path
        data_file = pd.read_csv(file_path, delim_whitespace=True)
        mean_wvls = 1. / data_file['wvn[um^-1]'].mean()
        wvl0 = float(self.args['wvl'])
        
        if wvl_check and np.absolute(wvl0 - mean_wvls) > 0.1:
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
                         xmin=0.5,
                         xmax=3.0,
                         savefig=False):
    """ Plot spectral directional-hemispherical reflectance for a given
        shape habit and roughness.
    """
    plt.style.use('agu_quarter')
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
    plt.xlim((xmin, xmax))
    #plt.xticks(np.arange(xmin, xmax + 0.1, 0.2))
    plt.ylim((0.0,1.0))
    #plt.yticks(np.arange(0, 1.1, 0.1))
    
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
    
def get_Lambertian_nlevels(Lambertian_file_path, theta_bins, phi_bins=None):
    """ Read data from Lambertian_file_path and calculate weighted standard 
        deviation of BRFs of a Lambertian surface to use to calculate nlevels 
        with relative precision
            
        Returns nlevels for BRF contourf plot
    """
    (shape_dir, Lambertian_file) = os.path.split(Lambertian_file_path)
    (top_data_dir, shape) = os.path.split(shape_dir)

    wvl = Lambertian_file.split('_')[0]
    half_width = Lambertian_file.split('_')[1]
    n_photon = Lambertian_file.split('_')[3]
    theta_0 = Lambertian_file.split('_')[4]
    
    LambertianData = MonteCarloDataSet(top_data_dir=top_data_dir,
                                       shapes=[shape],
                                       roughnesses=['smooth'],
                                       wvl=wvl,
                                       half_width=half_width,
                                       n_photon=n_photon,
                                       theta_0=theta_0)
                                       
    (albedo,
     mean_wvls) = LambertianData.directional_hemispherical_reflectance(
                                                        Lambertian_file_path)
    if phi_bins == None:
        phi_bins = 4 * theta_bins
    (brf,
    midpoints,
    mean_wvls) = LambertianData.bi_directional_reflectance_factor_3D(
                                                        Lambertian_file_path,
                                                        phi_bins,
                                                        theta_bins)
                                                        
    weights = ((np.sin(midpoints[1]) * np.cos(midpoints[1])) /
               np.sum((np.sin(midpoints[1]) * np.cos(midpoints[1]))))
               
    #print weights
    mean_brf = np.average(brf, axis=0, weights=weights).mean()
    
    azimuthal_vars = list()
    for phi_i, brf_vals in enumerate(brf.T):
        azimuthal_vars.append(np.sum(weights * (brf_vals - mean_brf)**2))
    
    std_brf = np.sqrt(np.mean(azimuthal_vars))
    nlevels = int(1. / std_brf)
    
    print('STD = %r' % std_brf)
    print('nlevels = %d' % nlevels)
    
    #return brf, midpoints
    return LambertianData
        
def calculate_bins(active_area, d_dome):
    """ Calculate number of bins to simulate photodiode with given active area
        mounted in dome with given diameter

        Returns number of bins n_bins
    """
    n_bins =  int((np.pi * d_dome) / (4.*active_area))
    print('BRF computing over %d theta bins' % n_bins)
    
    return n_bins

class UpperHammerAxes(HammerAxes):
    name = 'upper_hammer'
    def cla(self):
        HammerAxes.cla(self)
        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, 0, np.pi / 2.0)

    def _gen_axes_patch(self):
        #return Wedge((0.5, 0.5), 0.5, 180, 360)
        return Wedge((0.5, 0.5), 0.5, 0, 180)
        
    def _gen_axes_spines(self):
        path = Wedge((0, 0), 1.0, 0, 180).get_path()
        spine = mspines.Spine(self, 'circle', path)
        spine.set_patch_circle((0.5, 0.5), 0.5)
        return {'wedge':spine}

class UpperAitoffAxes(AitoffAxes):
    name = 'upper_aitoff'
    def cla(self):
        AitoffAxes.cla(self)
        Axes.set_xlim(self, -np.pi, np.pi)
        Axes.set_ylim(self, 0, np.pi / 2.0)

    def _gen_axes_patch(self):
        #return Wedge((0.5, 0.5), 0.5, 180, 360)
        return Wedge((0.5, 0.5), 0.5, 0, 180)
        
    def _gen_axes_spines(self):
        path = Wedge((0, 0), 1.0, 0, 180).get_path()
        spine = mspines.Spine(self, 'circle', path)
        spine.set_patch_circle((0.5, 0.5), 0.5)
        return {'wedge':spine}

def show_UpperHammerAxes():
    mprojections.register_projection(UpperHammerAxes)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='upper_hammer')
    ax.grid(True)
    
    theta = np.deg2rad(np.arange(0, 90))
    phi = np.deg2rad(np.arange(0, 360)) - np.pi
    x = np.empty((theta.size, phi.size))
    
    for i, theta_rad in enumerate(theta):
        for j, phi_rad in enumerate(phi):
            x[i,j] = theta_rad
    
    cax = ax.contourf(phi, theta[::-1], x)
    
    ax.grid(True)
    cb = fig.colorbar(cax, orientation='horizontal')
          
    plt.show()
    
def show_UpperAitoffAxes():
    mprojections.register_projection(UpperAitoffAxes)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='upper_aitoff')
    ax.grid(True)
    
    theta = np.deg2rad(np.arange(0, 90))
    phi = np.deg2rad(np.arange(0, 360)) - np.pi
    x = np.empty((theta.size, phi.size))
    
    for i, theta_rad in enumerate(theta):
        for j, phi_rad in enumerate(phi):
            x[i,j] = theta_rad
    
    cax = ax.contourf(phi, theta[::-1], x)
    
    ax.grid(True)
    cb = fig.colorbar(cax, orientation='horizontal')
          
    plt.show()

def nerd_ssa_cal_30(savefig=True):    
    """ SSA data from micro-CT (m^2/kg) and 0-30 1.3um BRFs from NERD
    """
    
    wvl13 = MonteCarloDataSet(shapes=['sphere','droxtal', 
                                      'solid_hexagonal_column',
                                      ], 
                              roughnesses=['smooth'])
                              
    wvl13.add_observational_data(8842.11/DENSITY_ICE,
                                 [0.25, 0.26, 0.25, 0.25],
                                 '2016c clear bin')
    wvl13.add_observational_data(8673.06/DENSITY_ICE,
                                 [0.22, 0.22, 0.20, 0.21],
                                 '2016e crystal chamber')
    wvl13.add_observational_data(15945.00/DENSITY_ICE,
                                 [0.38, 0.38, 0.39, 0.39],
                                 '2016b hotel sample')
    wvl13.add_observational_data(18868.39/DENSITY_ICE,
                                 [0.42, 0.43, 0.43, 0.42],
                                 '2016a John sample')
    wvl13.add_observational_data(8567.44/DENSITY_ICE,
                                 [0.27, 0.28, 0.28, 0.28],
                                 '2016d trash can')
    
    wvl13.add_observational_data(42122.25/DENSITY_ICE,
                                 [0.547, 0.567, 0.525, 0.527],
                                 '08:00EST 2-17-17 natural')
    wvl13.add_observational_data(32712.86/DENSITY_ICE,
                                 [0.469, 0.478, 0.480, 0.479],
                                 '13:00EST 2-17-17 natural')
    wvl13.add_observational_data(44232.67/DENSITY_ICE,
                                 [0.522, 0.527, 0.531, 0.524],
                                 '08:00EST 2-17-17 sand')
    wvl13.add_observational_data(15320.55/DENSITY_ICE,
                                 [0.182, 0.185, 0.240, 0.235],
                                 '13:00EST 2-17-17 sand')
    wvl13.add_observational_data(40592.91/DENSITY_ICE,
                                 [0.513, 0.515, 0.517, 0.519],
                                 '08:00EST 2-17-17 BC')
    wvl13.add_observational_data(21236.38/DENSITY_ICE,
                                 [0.287, 0.296, 0.455, 0.456],
                                 '13:00EST 2-17-17 BC')
    wvl13.add_observational_data(39368.58/DENSITY_ICE,
                                 [0.572, 0.577, 0.570, 0.566],
                                 '07:00EST 2-10-17 natural')
    wvl13.add_observational_data(37689.36/DENSITY_ICE,
                                 [0.601, 0.603, 0.563, 0.565],
                                 '15:00EST 2-10-17 natural')
    wvl13.add_observational_data(40333.84/DENSITY_ICE,
                                 [0.517, 0.517, 0.517, 0.514],
                                 '07:00EST 2-10-17 BC')
    
    wvl13.plot_bidirectional_reflectance_factor(30, theta_bins=18,
                                                overlay_nerd_obs=True,
                                                savefig=savefig)
                                                
def nerd_ssa_cal_60(savefig=False):
    wvl13 = MonteCarloDataSet(shapes=['sphere','droxtal', 
                                      'solid_hexagonal_column'], 
                              roughnesses=['smooth'])
                              
    wvl13.add_observational_data(8842.11/DENSITY_ICE,
                                 [0.29, 0.31, 0.31, 0.28],
                                 '2016c clear bin')
    wvl13.add_observational_data(8673.06/DENSITY_ICE,
                                 [0.28, 0.27, 0.26, 0.24],
                                 '2016e crystal chamber')
    wvl13.add_observational_data(15945.00/DENSITY_ICE,
                                 [0.41, 0.42, 0.43, 0.41],
                                 '2016b hotel sample')
    wvl13.add_observational_data(18868.34/DENSITY_ICE,
                                 [0.43, 0.49, 0.45, 0.40],
                                 '2016a John sample')
    wvl13.add_observational_data(8567.44/DENSITY_ICE,
                                 [0.34, 0.34, 0.36, 0.33],
                                 '2016d trash can')
    wvl13.add_observational_data(42122.25/DENSITY_ICE,
                                 [0.500, 0.565, 0.532, 0.545],
                                 '08:00EST 2-17-17 natural')
    wvl13.add_observational_data(32712.86/DENSITY_ICE,
                                 [0.49, 0.50, np.nan, np.nan],
                                 '13:00EST 2-17-17 natural')
    wvl13.add_observational_data(44232.67/DENSITY_ICE,
                                 [0.548, 0.544, 0.534, 0.507],
                                 '08:00EST 2-17-17 sand')
    wvl13.add_observational_data(15320.55/DENSITY_ICE,
                                 [0.235, 0.230, 0.311, 0.292],
                                 '13:00EST 2-17-17 sand')
    wvl13.add_observational_data(40592.91/DENSITY_ICE,
                                 [0.544, 0.531, 0.526, 0.522],
                                 '08:00EST 2-17-17 BC')
    wvl13.add_observational_data(21236.38/DENSITY_ICE,
                                 [0.325, 0.351, 0.481, 0.487],
                                 '13:00EST 2-17-17 BC')
    wvl13.add_observational_data(39368.58/DENSITY_ICE,
                                 [0.588, 0.587, 0.598, 0.584],
                                 '07:00EST 2-10-17 natural')
    wvl13.add_observational_data(37689.36/DENSITY_ICE,
                                 [0.642, 0.623, 0.61, 0.62],
                                 '15:00EST 2-10-17 natural')
    wvl13.add_observational_data(40333.84/DENSITY_ICE,
                                 [0.539, 0.534, 0.535, 0.537],
                                 '07:00EST 2-10-17 BC')
                                 
    wvl13.plot_bidirectional_reflectance_factor(60, theta_bins=18,
                                                overlay_nerd_obs=True,
                                                savefig=savefig)

def main():
    #data = MonteCarloDataSet()
    #data.plot_directional_hemispherical_reflectance()
    #show_UpperHammerAxes()
    plt.style.use('agu_quarter')
    plt.style.use('grl')
    nerd_ssa_cal_30(savefig=True)
    nerd_ssa_cal_60(savefig=True)

if __name__=='__main__':
    main()
