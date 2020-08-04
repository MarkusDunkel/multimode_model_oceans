#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 19:20:59 2020

@author: markus
"""
import os
import shutil
import netCDF4
import pickle
import numpy as np

class OutputCompilation():
    def __init__(self, o):
        self.shape = (o.N_y, o.N_x, o.N_z)
        self.mode_num = o.mode_num 
        self.mode_list = o.mode_list
        self.save_as = o.save_as
        self.x = o.x
        self.y = o.y
        self.save_time_steps = o.save_time_steps
        self.zon_sect = o.zon_sect
        self.mer_sect = o.mer_sect
        self.variable_names = o.variable_names
        self.av_window = o.av_window
        self.dt = o.dt
        
        path = '../' + self.save_as
        try:
            shutil.rmtree(path)
        except: None
        os.mkdir(path)
        self.path = path + '/'
        
#==============================================================================
#------------------------------ Settings by User ------------------------------
 
        self.file_names = {
            'swm_fields':     ('time','mode','y','x'),
            'zon_sect':       ('time','mode','y','x'), 
            'mer_sect':       ('time','mode','y','x'),  
            }
        
        self.time = {
            'swm_fields': [None], 
            'zon_sect':   [None], 
            'mer_sect':   [None],  
            }
        
        self.mode = {
            'swm_fields': self.mode_list, 
            'zon_sect':   self.mode_list, 
            'mer_sect':   self.mode_list,  
            }
        
        self.y = {
            'swm_fields': self.y, 
            'zon_sect':   self.zon_sect, 
            'mer_sect':   self.y, 
            }
        
        self.x = {
            'swm_fields': self.x, 
            'zon_sect':   self.x, 
            'mer_sect':   self.mer_sect, 
            }
        
        self.continous_rep = ('zon_sect', 'mer_sect')
        self.sample_rep = ('swm_fields',)
        
                     
        dim_units = {'time':'seconds', 'mode':'mode', 'y':'m north', 
                     'x':'meters east', 'z': 'depth [m]'}
        
        var_units = {'u':'m/s', 'v':'m/s', 'w':'m/s', 'h':'m',
                     'vmix_u':'1', 'vmix_v':'1', 'vmix_h':'1',
                     'nl_u':'1', 'nl_v':'1', 'nl_h':'1'}
        
#------------------------------- Settings by User -----------------------------
#============================================================================== 
        
        
        pickle.dump( o.__dict__, open( self.path + self.save_as, "wb" ) )
        
        for name in self.file_names.keys():
            
            ncfile = netCDF4.Dataset(
                self.path + name, mode='w', format='NETCDF4_CLASSIC')
            ncfile.close()
            os.chmod(self.path + name, 0o0777)
            ncfile = netCDF4.Dataset(self.path + name, mode='w')
            
            dim_names = self.file_names[name]
            for dim in dim_names:
                if self.__dict__[dim][name][0] is not None:
                    ncfile.createDimension(dim, len(self.__dict__[dim][name]))
                else:
                    ncfile.createDimension(dim, None)
                ncfile.createVariable(dim, np.float32, (dim,))
                handle = ncfile.variables[dim]
                handle.units = dim_units[dim]
                handle.long_name = dim 
                if self.__dict__[dim][name][0] is not None:
                    handle[:] = self.__dict__[dim][name]
                    
            for var in self.variable_names:
                ncfile.createVariable(var, np.float32, dim_names)
                handle = ncfile.variables[var]
                handle.units = var_units[var]
    
            ncfile.close()
        
        
    def write_to_ncfile(self, t, fields, files):
        
        for name in files:
            ncfile = netCDF4.Dataset(self.path + name, mode='r+')
            indx = len(ncfile['time'])
            ncfile['time'][indx] = ( t - int(self.av_window/2) ) * self.dt
           
            if name == 'swm_fields':
                for var in self.variable_names:
                    for m in range(self.mode_num):
                        ncfile[var][indx, m, :, :] = np.expand_dims(
                            fields[var].data[m], axis=0 )
              
            elif name == 'zon_sect':
                for var in self.variable_names:
                    for m in range(self.mode_num):
                        ncfile[var][indx, m, :, :] = np.expand_dims( 
                            fields[var].data[m][self.zon_sect,:], axis=0 )
                       
            elif name == 'mer_sect':
                for var in self.variable_names:
                    for m in range(self.mode_num):
                        ncfile[var][indx, m, :, :] = np.expand_dims( 
                            fields[var].data[m][:, self.mer_sect], axis=0 )
    
            ncfile.close()
            
            

        
  
        
        
        
   