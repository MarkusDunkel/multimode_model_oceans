#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:49:38 2020
@author: Markus Dunkel
"""
import numpy as np

class ThreeD:
    """Class to store and manipulate 3D variables. 3D variables are in general 
    mode independent. 
    """
    def __init__( self, arr = np.zeros((0,0)) ):
        
        self.array = arr
        self.shape = arr.shape
        
    def TwoD_to_ThreeD(self, TwoD_object, modes):
        
        data = np.stack(TwoD_object.data, axis=0)
        self.array = np.tensordot(data, modes, axes=[0, 0])
        
        
class TwoD:
    """Class to store and manipulate 2D variables. 2D variables are in general 
    mode dependent.
    """
    def __init__(self, array, mode_num, halo):
        self.mode_num = mode_num 
        self.data=[]
        # always with halos
        self.shape = np.shape(array)
        for i in range( self.mode_num ):
            self.data.append( array.copy() )
            
            
        self._local_slice = ( 
            slice(halo[0], -halo[0]), slice(halo[1], -halo[1]) )
            
    def gom(self, mode): 
        """
        Get a single mode
        """
        return self.data[mode]
    
        
    def get_all_modes(self, index = False):
        """
        Get view of all modes with frame [index]. Output is a list of numpy 
        arrays.
        """
        if index:
            out = []
            for i in range(self.mode_num):
                out.append( self.data[i][index] )
        else:
            out = self.data
        return out
        
    
    def wom(self, mode, array):
        """ Writes to a single mode with position [mode]."""
        self.data[mode][:] = array
        
        
    def write_to_all_modes(self, array, index = False):
        """Writes to all modes the same array if input array is a single field
        or writes to each mode a different one if input array is a list of 
        fields with equal length as self.mode_num. 
        """
        
        if index:
            if isinstance(array, list):
                 for m in range(self.mode_num):
                    self.data[m][index] = array[m]
            else:
                for m in range(self.mode_num):
                    self.data[m][index] = array
                    
        else:
            if isinstance(array, list):
                 for m in range(self.mode_num):
                    self.data[m][:] = array[m]
            else:
                for m in range(self.mode_num):
                    self.data[m][:] = array
                    
    def add_to_all_modes(self, array, index = False):
        """Writes to all modes the same array if input array is a single field
        or writes to each mode a different one if input array is a list of 
        fields with equal length as self.mode_num. 
        """
        
        if index:
            if isinstance(array, list):
                 for m in range(self.mode_num):
                    self.data[m][index] += array[m]
            else:
                for m in range(self.mode_num):
                    self.data[m][index] += array
                    
        else:
            if isinstance(array, list):
                 for m in range(self.mode_num):
                    self.data[m][:] += array[m]
            else:
                for m in range(self.mode_num):
                    self.data[m][:] += array
        
        
    def receive_flatten_array(self, data, indx):
        """Writes to a flatten view of all modes. Array can be a list with 
        equal length of mode number or a single array.
        """
        if isinstance(data, list):
            for m in range(self.mode_num):
                self.data[m].reshape((-1,))[indx] = data[m]
        else: 
            for m in range(self.mode_num):
                self.data[m].reshape((-1,))[indx] = data
               
        
    def get_flatten_array(self, _local_slice, indx):
        """Gets a flatten view of arrays in list."""
        out=[]
        for m in range(self.mode_num):
            out.append( self.data[m][_local_slice].reshape((-1,))[indx] )
            
        return out
            
class Profile:
    """ This class is for all vertical profile variables and adds 
    functionality such as switching between w and p levels."""
    
    def __init__(self, mode_num, N_z, array, init_level, mode_dependent):
        
        self.mode_num = mode_num
        self.N_z = N_z
        #----------------------------------------------------------------------
        #shaping input array into right format
        
        if init_level == 'w':
            level_num = self.N_z
        elif init_level == 'p':
            level_num = self.N_z - 1
        else: print('wrong usage of init_level!')
        
        
        """Format of data stored in this class - list of 1-dim array with 
        length 1 if profile is independent of modes and length of mode number 
        if profile is mode-dependent. 
        """
        
        if mode_dependent:
            variable_format = []
            for m in range(self.mode_num):
                variable_format.append( np.zeros((level_num)) )
        else:
            variable_format = [np.zeros((level_num))]
        
        
        """Testing of input format and writing to correct format."""
        arr_shape = np.shape(array)
        
        if arr_shape.count(self.mode_num) == 1 and mode_dependent:
            if isinstance(array, list):
                for m in range(len(variable_format)):
                    variable_format[m] = array[m][:]
            elif arr_shape[0] == self.mode_num:
                for m in range(len(variable_format)):
                    variable_format[m] = array[m,:]
            elif arr_shape[1] == self.mode_num:
                for m in range(len(variable_format)):
                    variable_format[m] = array[:,m]
                
        elif arr_shape.count(self.mode_num) != 1 and mode_dependent == False:
            variable_format[0][:] = array[:]
            
        else:
            print("Input data not appropriate for initializing profile \
                  variable. This error message is also raised if number of \
                  depth levels equals number of modes.")
                  
        #final testing if data is in correct format 
        list_len = len(variable_format)
        assert list_len == 1 or list_len == self.mode_num
        
        for m in range(list_len):
            assert len(variable_format[m]) == level_num
            
        #----------------------------------------------------------------------
        # initialization of object of class Profile
        
        if init_level == 'w' and mode_dependent:
            self._on_w = variable_format
            self._on_p = self.interpolate_to_p_levels( variable_format )
            
        elif init_level == 'w' and not mode_dependent:
            self._on_w = variable_format[0]
            dummy = self.interpolate_to_p_levels( variable_format )
            self._on_p = dummy[0]
            
        elif init_level == 'p' and mode_dependent:
            self._on_p = variable_format
            self._on_w = self.interpolate_to_w_levels( variable_format )
            
        elif init_level == 'p' and not mode_dependent:
            self._on_p = variable_format[0]
            dummy = self.interpolate_to_w_levels( variable_format )
            self._on_w = dummy[0]
            
        
    def interpolate_to_p_levels(self, data):
        "Interpolates profile to p levels if it is on w levels initially."
        array = [ np.zeros(( len(data[0]) -1 )) ] * len(data)
        for m in range(len(data)):
            array[m] = ( data[m][:-1] + data[m][1:] ) /2
            
        return array
        
    def interpolate_to_w_levels(self, data):
        "Interpolates profile to w levels if it is on p levels initially."
        array = [ np.zeros(( len(data[0]) +1 )) ] * len(data)
        for m in range(len(data)):
            arr_extended = np.zeros(( len(data[0]) + 2 ))
            arr_extended[1:-1] = data[m][:]
            arr_extended[0] = data[m][0]
            arr_extended[-1] = data[m][-1]
            array[m] = ( arr_extended[:-1] + arr_extended[1:] ) /2
        
        return array
        
        
        
        
        