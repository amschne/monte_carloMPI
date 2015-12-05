#!/usr/bin/env python
"""
"""
import os
import argparse
import fnmatch

import numpy as np
import pandas as pd

def get_args():
    """ User input
    """
    top_data_dir = '/data3/amaschne/AGU2015'
    shapes = ['sphere',
              'solid_hexagonal_column', 
              'hollow_hexagonal_column',
              'hexagonal_plate',
              'droxtal',
              'hollow_bullet_rosette',
              'solid_bullet_rosette',
              '8-element_column_aggregate',
              '5-element_plate_aggregate',
              '10-element_plate_aggregate']
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
        
        def plot_directional_hemispherical_reflectance(self):
            """ Plot directional-hemispherical reflectance as a function of
                particle effective radius
            """
            fig = plt.figure()
            wvl_nm = np.around(float(self.args['wvl']) * 10**3)
            
            num_shapes = len(self.args['shapes'])
            num_colors = num_shapes
            for i, shape in self.args['shapes']:
                if shape == 'sphere':
                    num_colors = num_colors - 1
            
            color_list = plt.cm.Dark2(np.linspace(0, 1, num_colors))
            for i, shape in self.args['shapes']:
                print('Calculating and plotting albedo for %ss...' % shape)
                
                label = re.sub(r'[\W_]', ' ', shape)
                
                if shape == 'sphere':
                    # Full scattering phase functions
                    particle_radii = list()
                    albedo = list()
                    for RE, file_path in self.data_I[shape].items()
                        particle_radii.append(float(RE))
                        albedo.append(
                          self.directional_hemispherical_reflectance(file_path))
                    
                    plt.plot(particle_radii, albedo, color='k', marker='o')
                    
                    # Henyey Greenstein scattering phase functions
                    particle_radii = list()
                    albedo = list()
                    for RE, file_path in self.data_HG[shape].items()
                        particle_radii.append(float(RE))
                        albedo.append(
                          self.directional_hemispherical_reflectance(file_path))
                    
                    plt.plot(particle_radii, albedo, color='k', marker='o',
                             linestyle='dashed', label=label)
                else:
                    for j, roughness in self.args['roughness']:
                        if roughness == 'smooth':
                            marker = 'o'
                        elif roughness == 'moderately rough':
                            marker = 'd'
                        elif roughness == 'severely rough':
                            marker = '*'
                        
                        # Full scattering phase functions
                        particle_radii = list()
                        albedo = list()
                        for RE, file_path in self.data_I[shape].items()
                            particle_radii.append(float(RE))
                            albedo.append(
                                     self.directional_hemispherical_reflectance(
                                                                     file_path))
                        plt.plot(particle_radii, albedo, color=color_list[i],
                                 marker=marker)
                    
                        # Henyey Greenstein scattering phase functions
                        particle_radii = list()
                        albedo = list()
                        for RE, file_path in self.data_HG[shape][
                                                              roughness].items()
                            particle_radii.append(float(RE))
                            albedo.append(
                                     self.directional_hemispherical_reflectance(
                                                                     file_path))
                    
                        plt.plot(particle_radii, albedo, color=color_list[i],                                    marker=marker, linestyle='dashed', 
                                 label=label)
            
            plt.xlabel('Ice particle effective radius ($\mu m$)')
            plt.ylabel('Reflectance')
            plt.title('Nadir directional-hemispherical reflectance for ' 
                      '$\lambda_0$ = %snm' % wvl_nm)
            plt.legend()
            plt.grid()
            
            plt.show()
            
        def directional_hemispherical_reflectance(self, file_path):
            """ Read in data and calculate directional-hemispherical 
                reflectance, a.k.a. black sky albedo
            """
            data_file = pd.read_csv(file_path, delim_whitespace=True)
            mean_wvls = 1. / data['wvn[um^-1]'].mean()
            wvl0 = float(self.args['wvl'])
            
            if np.absolute(wvl0 - mean_wvls) > 0.1:
                albedo = None
            else:
                Q_down = data['wvn[um^-1]'].sum()
                Q_up = data[data.condition==1]['wvn[um^-1]'].sum()
            
            albedo = Q_up / Q_down
            
            return albedo
            
            
            

def main():
    data = MonteCarloDataSet()
    data.plot_directional_hemispherical_reflectance()

if __name__=='__main__':
    main()