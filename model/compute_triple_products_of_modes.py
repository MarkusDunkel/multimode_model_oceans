#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:53:41 2020

@author: Markus Dunkel
"""

import numpy as np
from itertools import product
import variables as var


def compute_triple_tensor(mode_num, comb, prod):
    
    ten = np.zeros(( mode_num, mode_num, mode_num ))
    
    for i in range(len(comb)):
        ten[comb[i]] = prod[i]
        
    return ten


def z_derivative(arr, dz):
    # dz < 0, positive derivative is in upward direction
    d_arr = np.diff(arr)
    arr_z = -d_arr / dz
    
    return arr_z


def product_of_modes_3( prod, comb, mode_num, dz, modes ):
   
    for i in range(len(comb)):
        for j in range(1):
            prod[i] *= modes[j][comb[i][j]]
            
        prod[i] = sum( prod[i] ) * dz 
               
    return prod


def product_of_modes_2( comb, mode_num, dz, modes ):
    
    prod = [0] * len(comb)
    arr = np.ones(len(modes[-1][0]))
    
    for i in range(len(comb)):
        for j in range(1, 3):
            arr *= modes[j][comb[i][j]]
            
        arr = z_derivative( arr, dz )
        
        prod[i] = arr
        arr = np.ones(len(modes[-1][0]))  
        
    return prod


def product_of_modes_1( comb, mode_num, dz, modes ):
    
    prod = [0] * len(comb)
    arr = np.ones(len(modes[0][0]))
    for i in range(len(comb)):
        for j in range(3):
            arr *= modes[j][comb[i][j]]
            
        prod[i] += sum ( arr ) * dz
        arr = np.ones(len(modes[0][0]))
             
    return prod
    
    
def compute_triple_mode_tensors(
        pmodes, wmodes, mode_num, dz_int, N_z, Nsq, rho_0, g):
    
    density_z = var.Profile( 
        mode_num, N_z, Nsq._on_w * rho_0 * (-1) / g, 'w', False) 
    
    density_zz = var.Profile( 
        mode_num, N_z, z_derivative( density_z._on_w, dz_int ), 'p', False)
    
    comb = list( product(range(mode_num), repeat = 3) )
    
#------------------------------------------------------------------------------
    #compute PPP
    triple_modes = (pmodes._on_p, pmodes._on_p, pmodes._on_p)
    
    prod = product_of_modes_1( comb, mode_num, dz_int, triple_modes )
    
    PPP = compute_triple_tensor( mode_num, comb, prod )
    
#------------------------------------------------------------------------------
    #compute P(PW)_z
    triple_modes = (pmodes._on_p, pmodes._on_w, wmodes._on_w)
    
    prod = product_of_modes_2( comb, mode_num, dz_int, triple_modes )
    
    prod = product_of_modes_3( prod, comb, mode_num, dz_int, triple_modes)
       
    PPWz = compute_triple_tensor( mode_num, comb, prod )
    
#------------------------------------------------------------------------------
    # compute WPW_dz
    triple_modes = (wmodes._on_p * density_z._on_p, pmodes._on_p, wmodes._on_p)
    
    prod = product_of_modes_1( comb, mode_num, dz_int, triple_modes )
    
    WPWdz = compute_triple_tensor( mode_num, comb, prod )
    
#------------------------------------------------------------------------------
    #compute W(WW)z_dz
    triple_modes = (wmodes._on_p * density_z._on_p, wmodes._on_w, wmodes._on_w)
    
    prod = product_of_modes_2( comb, mode_num, dz_int, triple_modes )
    
    prod = product_of_modes_3( prod, comb, mode_num, dz_int, triple_modes)
    
    WWWzdz = compute_triple_tensor( mode_num, comb, prod )

#------------------------------------------------------------------------------
    #compute  WWW_dzz
    triple_modes= (wmodes._on_p * density_zz._on_p, wmodes._on_p, wmodes._on_p)
    
    prod = product_of_modes_1( comb, mode_num, dz_int, triple_modes )
    
    WWWdzz = compute_triple_tensor( mode_num, comb, prod )
    
#------------------------------------------------------------------------------
    
    return (PPP, PPWz, WPWdz, WWWzdz, WWWdzz)
    
    
    
    
    