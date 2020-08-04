#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 13:37:13 2020

@author: Markus Dunkel
"""
import numpy as np
import variables as var
from numba import jit

#------------------------------------------------------------------------------

def vertical_integration( vert_diff, modes, mode_num, coef ):
    
    nzz = np.tensordot( vert_diff, modes, axes=[-1, -1] ) 
    
    out = []
    for i in range(mode_num):
        out.append(nzz[:,:,i] * coef[i])
        
    return out 

#------------------------------------------------------------------------------
    
#parallized z-derivative
@jit(nopython=True)
def z_derivative(mat, out, shape, coef):
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]-1):
                out[i, j, k] = mat[i, j, k] - mat[i, j, k+1]
                out[i, j, k] *= coef #*(-1)
    return out
                
#------------------------------------------------------------------------------
    
#apply boundary condition
@jit(nopython=True)
def apply_boundary_condition(mat, shape, halo, tau_x, tau_y, rho_0, name):      
        
    if name == 'momentum_u':
        boundary = np.zeros((shape[0], shape[1], 1))
        boundary[halo[0]:-halo[0], halo[1]:-halo[1],0] = tau_x / rho_0
        mat = np.concatenate((boundary, mat), axis=2 ) 
   
    if name == 'momentum_v':
        boundary = np.zeros((shape[0], shape[1], 1))
        boundary[halo[0]:-halo[0], halo[1]:-halo[1],0] = tau_y / rho_0
        mat = np.concatenate((boundary, mat), axis=2 )
        
    if name != 'momentum_u' and name !='momentum_v':
        boundary = np.zeros((shape[0], shape[1], 1))
        mat = np.concatenate((mat, boundary), axis=2 ) 
           
    return mat

#------------------------------------------------------------------------------
def vertical_diffusivity( state ):
    
    mat, coef, rho_0, dz, halo, shape, tau_x, tau_y, name = state
    
    out = np.zeros((shape[0], shape[1], shape[2]-1))
    mat = z_derivative( mat, out, shape, 1/dz * coef)
    
    mat = apply_boundary_condition(mat, shape, halo, tau_x, tau_y, rho_0, name)
       
    mat = z_derivative( mat, out, shape, 1/dz )
     
    boundary = np.zeros((shape[0], shape[1], 1))
    mat = np.concatenate((mat, boundary), axis=2 ) 
    
    return mat

#------------------------------------------------------------------------------
    
def compute_mixing_momentum(state):
    o, tau_x, tau_y, shape, halo, vel_swm, name = state
    
    vel = var.ThreeD() 
    vel.TwoD_to_ThreeD(vel_swm, o.pmodes._on_p)
    
    diff_vel = vertical_diffusivity( 
        (vel.array, o.nu, o.rho_0, o.dz_int, halo, np.shape(vel.array),
         tau_x, tau_y, name))
   
    nzz_vel = vertical_integration( 
        diff_vel, o.pmodes._on_p, o.mode_num, np.ones(
            (o.mode_num)) * o.dz_int )
    
    return nzz_vel

#------------------------------------------------------------------------------
        
def compute_mixing_density(state):
    o, shape, halo, h_swm = state
    
    rho_dash = var.ThreeD( np.zeros(shape) ) 
    rho_dash.TwoD_to_ThreeD( 
        h_swm, o.wmodes._on_w * o.Nsq._on_w * o.rho_0 / o.g )
    
    diff_h = vertical_diffusivity(( rho_dash.array, o.kap, o.rho_0, o.dz_int,
                                   halo, np.shape(rho_dash.array),
                                   o.tau_x, o.tau_y, 'density'))
    
    nzz_h = vertical_integration( 
        diff_h, o.wmodes._on_w, o.mode_num, o.H / o.rho_0 * o.dz_int )
    
    return nzz_h
        
    
    