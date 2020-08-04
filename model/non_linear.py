#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 16:18:13 2020

@author: Markus Dunkel
"""
import numpy as np
from itertools import combinations_with_replacement
from itertools import product
from numba import jit

#------------------------------------------------------------------------------
#shift velocity matrices on grid; 
#e.g. uonvy is u averaged such that it is aligned with v in y direction. 
@jit(nopython=True)
def shift_fields_u( mode_num, N_y, N_x, un, vn, wn ):
    
    uonvy = np.zeros(( mode_num, N_y, N_x ))
    uonvx = np.zeros(( mode_num, N_y, N_x ))
    vonuy = np.zeros(( mode_num, N_y, N_x ))
    wonu =  np.zeros(( mode_num, N_y, N_x ))
    
    for m in range(mode_num):
        
        for j in range(1, N_y):
            for i in range(N_x-1):
                uonvy[m, j, i]   = (un[m, j, i] + un[m, j, i+1]) / 2  
                uonvx[m, j, i]   = (un[m, j-1, i] + un[m, j, i]) / 2  
                vonuy[m, j, i+1] = (vn[m, j, i] + vn[m, j, i+1]) / 2  
                wonu[m, j, i+1]  = (wn[m, j, i] + wn[m, j, i+1]) / 2  
       
    return uonvy, uonvx, vonuy, wonu

#------------------------------------------------------------------------------
    
@jit(nopython=True)    
def shift_fields_v( mode_num, N_y, N_x, un, vn, wn ):
    
    uonvx = np.zeros(( mode_num, N_y, N_x ))
    vonuy = np.zeros(( mode_num, N_y, N_x )) 
    vonux = np.zeros(( mode_num, N_y, N_x ))
    wonv = np.zeros(( mode_num, N_y, N_x ))
    
    for m in range(mode_num):
        
        for j in range(1, N_y):
            for i in range(N_x-1):
                vonuy[m, j, i+1] = (vn[m, j, i] + vn[m, j, i+1]) / 2 
                uonvx[m, j, i]   = (un[m, j-1, i] + un[m, j, i]) / 2  
                vonux[m, j-1, i] = (vn[m, j-1, i] + vn[m, j, i]) / 2 
                wonv[m, j, i]    = (wn[m, j-1, i] + wn[m, j, i]) / 2 
    
    return vonuy, uonvx, vonux, wonv

#------------------------------------------------------------------------------

@jit(nopython=True)
def uu_x( mode_num, N_y, N_x, dx, halo, uonvy, indices ): 
    
    a = np.zeros(( mode_num, mode_num, N_y, N_x ))
    
    for k in range(len(indices)):
        e = indices[k, 0]
        f = indices[k, 1]
   
        for j in range(halo[0], N_y-halo[0]):
            for i in range(halo[1], N_x-halo[1]):        
                a[ e, f, j, i ] = (uonvy[e, j, i]*uonvy[f, j, i] -
                                   uonvy[e, j, i-1]*uonvy[f, j, i-1]) / dx
        
        a[f, e, :, :] = a[e, f, :, :]
          
    return a

#------------------------------------------------------------------------------
    
@jit(nopython=True)
def vv_y( mode_num, N_y, N_x, dy, halo, vonux, indices ): 
    
    a = np.zeros(( mode_num, mode_num, N_y, N_x ))
    
    for k in range(len(indices)):
        e = indices[k, 0]
        f = indices[k, 1]
   
        for j in range(halo[0], N_y-halo[0]):
             for i in range(halo[1], N_x-halo[1]):        
                 a[e, f, j, i] = (vonux[e, j-1, i]*vonux[f, j-1, i] -
                                  vonux[e, j, i]*vonux[f, j, i]) / dy
        
        a[f, e, :, :] = a[e, f, :, :]
    
    return a

#------------------------------------------------------------------------------

@jit(nopython=True)
def uv_y( mode_num, N_y, N_x, dy, halo, uonvx_n, vonuy_m, indices): 
    
    a = np.zeros(( mode_num, mode_num, N_y, N_x ))
    
    for k in range(len(indices)):
        e = indices[k, 0]
        f = indices[k, 1]
    
        for j in range(halo[0], N_y-halo[0]):
            for i in range(halo[1], N_x-halo[1]):        
                a[e, f, j, i] = (uonvx_n[e, j, i] * vonuy_m[f, j, i] -
                                 uonvx_n[e, j+1, i] * vonuy_m[f, j+1, i]) / dy
         
    return a

#------------------------------------------------------------------------------
 
@jit(nopython=True)
def uv_x( mode_num, N_y, N_x, dx, halo, uonvx_n, vonuy_m, indices ): 
    
    a = np.zeros(( mode_num, mode_num, N_y, N_x ))
    
    for k in range(len(indices)):
        e = indices[k, 0]
        f = indices[k, 1]
    
        for j in range(halo[0], N_y-halo[0]):
            for i in range(halo[1], N_x-halo[1]):        
                a[e, f, j, i] = (uonvx_n[e, j, i+1]*vonuy_m[f, j, i+1] -
                                 uonvx_n[e, j, i]*vonuy_m[f, j, i]) / dx
             
    return a

#------------------------------------------------------------------------------
    
@jit(nopython=True)
def uw_or_vw( mode_num, N_y, N_x, halo, vel, wn, indices ): #ok
    
    velw = np.zeros(( mode_num, mode_num, N_y, N_x ))
    
    for k in range(len(indices)):
        e = indices[k, 0]
        f = indices[k, 1]
        
        for j in range(halo[0], N_y-halo[0]):
            for i in range(halo[1], N_x-halo[1]):
                velw[e, f, j, i] = vel[e, j, i] * wn[f, j, i]
    
    return velw

#------------------------------------------------------------------------------
    
def vertical_expansion( mode_num, N_y, N_x, tens, *args ):
    
    out_list = []
    for mat in args:
        
        out = np.tensordot( tens, mat, axes=([1,2],[0,1]) )
        
        out_list.append( out )
        
    if len(out_list) == 1:
        return out
                     
    return out_list

#------------------------------------------------------------------------------

@jit(nopython=True)
def merging( coef, name, mode_num, N_y, N_x , *args ):
    
    arr = np.zeros(( mode_num, N_y, N_x ))
    
    for mat in args:
       arr += mat
       
    #factoring
    if name == 'u' or name == 'v':
        arr *= coef[0]
    elif name == 'h':
        for m in range(mode_num):
            arr[m] *= coef[m]
    
    out = []
    for m in range(mode_num):
        out.append( arr[m, :, :] )
        
    return out

#------------------------------------------------------------------------------
    
def compute_nonlinear_u(o, un, vn, wn):
    
    comb =np.array(list(combinations_with_replacement( range(o.mode_num), 2 ))) 
    # ordes does not matter
    per = np.array(list(product( range(o.mode_num), repeat=2 ))) 
    # order does matter
    
    un = np.asarray(un)
    vn = np.asarray(vn)
    wn = np.asarray(wn)
    
    mode_num, N_y, N_x = np.shape(un)
   
    uonvy, uonvx, vonuy, wonu =shift_fields_u(o.mode_num, N_y, N_x, un, vn, wn)
    
    uux = uu_x( o.mode_num, N_y, N_x, o.dx, o.halo, uonvy, comb )
    uvy = uv_y( o.mode_num, N_y, N_x, o.dy, o.halo, uonvx, vonuy, per )
    
    uw = uw_or_vw( o.mode_num, N_y, N_x, o.halo, un, wonu, per )
   
    uux_pl_uvy_PPP=vertical_expansion( o.mode_num, N_y, N_x, o.PPP, uux + uvy )
    uw_PPWz = vertical_expansion( o.mode_num, N_y, N_x, o.PPWz, uw )
   
    Fx = merging( (-1,), 'u', o.mode_num, N_y, N_x, uux_pl_uvy_PPP, uw_PPWz )
    
    return Fx

#------------------------------------------------------------------------------
    
def compute_nonlinear_v(o, un, vn, wn):
    
    comb =np.array(list(combinations_with_replacement( range(o.mode_num), 2 ))) 
    # ordes does not matter
    per = np.array(list(product(range(o.mode_num), repeat=2 ))) 
    # order does matter
    
    #converting lists to numpy
    un = np.asarray(un)
    vn = np.asarray(vn)
    wn = np.asarray(wn)
    
    mode_num, N_y, N_x = np.shape(un)
    
    vonuy, uonvx, vonux, wonv =shift_fields_v(o.mode_num, N_y, N_x, un, vn, wn)
   
    vvy = vv_y( o.mode_num, N_y, N_x, o.dy, o.halo, vonux, comb )
    
    uvx = uv_x( o.mode_num, N_y, N_x, o.dx, o.halo, uonvx, vonuy, per )
    vw = uw_or_vw( o.mode_num, N_y, N_x, o.halo, vn, wonv, per )
    
    uvx_pl_vvy_PPP =vertical_expansion( o.mode_num, N_y, N_x, o.PPP, uvx + vvy)
    vw_PPWz = vertical_expansion( o.mode_num, N_y, N_x, o.PPWz, vw )
    
    Fy = merging( (-1,), 'v', o.mode_num, N_y, N_x, uvx_pl_vvy_PPP, vw_PPWz )
    
    return Fy

#------------------------------------------------------------------------------

@jit(nopython=True)
def uh_x( mode_num, N_y, N_x, dx, halo, u_n, honu_m, indices): 
    
    a = np.zeros(( mode_num, mode_num, N_y, N_x ))
    
    for k in range(len(indices)):
        e = indices[k, 0]
        f = indices[k, 1]
    
        for j in range(halo[0], N_y-halo[0]):
            for i in range(halo[1], N_x-halo[1]):        
                a[e, f, j, i] = (u_n[e, j, i+1]*honu_m[f, j, i+1] -
                                 u_n[e, j, i]*honu_m[f, j, i]) / dx
          
    return a

#------------------------------------------------------------------------------

@jit(nopython=True)
def vh_y( mode_num, N_y, N_x, dy, halo, v_n, honv_m, indices): 
    
    a = np.zeros(( mode_num, mode_num, N_y, N_x ))
    
    for k in range(len(indices)):
        e = indices[k, 0]
        f = indices[k, 1]
    
        for j in range(halo[0], N_y-halo[0]):
            for i in range(halo[1], N_x-halo[1]):        
                a[e, f, j, i] = (v_n[e, j, i]*honv_m[f, j, i] -
                                 v_n[e, j+1, i]*honv_m[f, j+1, i]) / dy
          
    return a

#------------------------------------------------------------------------------
    
@jit(nopython=True)    
def shift_fields_h( mode_num, N_y, N_x, un, vn, hn ):
    
    honu = np.zeros(( mode_num, N_y, N_x ))
    honv = np.zeros(( mode_num, N_y, N_x )) 
    
    for m in range(mode_num):
        
        for j in range(1, N_y):
            for i in range(N_x-1):
                honu[m, j, i+1] = (hn[m, j, i] + hn[m, j, i+1]) / 2 
                honv[m, j, i] = (hn[m, j-1, i] + hn[m, j, i]) / 2  
    
    return honu, honv
    
#------------------------------------------------------------------------------
    
def compute_nonlinear_h(o, un, vn, hn, wn):

    per = np.array(list(product( range(o.mode_num), repeat=2 ))) 
    # order does matter
    
    un = np.asarray(un)
    vn = np.asarray(vn)
    hn = np.asarray(hn)
    wn = np.asarray(wn)
    
    mode_num, N_y, N_x = np.shape(un)
   
    honu, honv = shift_fields_h(o.mode_num, N_y, N_x, un, vn, hn)
    
    uhx = uh_x( o.mode_num, N_y, N_x, o.dx, o.halo, un, honu, per ) 
    vhy = vh_y( o.mode_num, N_y, N_x, o.dy, o.halo, vn, honv, per )
    
    wh = uw_or_vw( o.mode_num, N_y, N_x, o.halo, wn, hn, per )
   
    uhx_pl_vhy_WPWdz = vertical_expansion( 
        o.mode_num, N_y, N_x, o.WPWdz, uhx + vhy )
    wh_WWWzdz = vertical_expansion( o.mode_num, N_y, N_x, o.WWWzdz, wh )
    wh_WWWdzz = vertical_expansion( o.mode_num, N_y, N_x, o.WWWdzz, wh )
   
    Fh = merging( 
        o.H/o.rho_0, 'h', o.mode_num, N_y, N_x, uhx_pl_vhy_WPWdz, wh_WWWzdz,  
        wh_WWWdzz )
    
    return Fh

#------------------------------------------------------------------------------
    
@jit(nopython=True)
def vectorized_loop(mode_num, N_y, N_x, arr, sh, field):
    
    for m in range(mode_num):
        for j in range(1, N_y):
            for i in range(N_x-1):
                field[m,j+sh[0],i+sh[1]] = ( 
                arr[m,j+sh[2],i+sh[3]] + arr[m,j+sh[4],i+sh[5]] ) / 2
                
    return field
        

def shift_fields( mode_num, N_y, N_x, un, vn, hn, wn, fields ):
    
    for name in fields.keys():
        field = np.zeros(( mode_num, N_y, N_x ))
                        
        if name == 'uonvy':
            arr = un
            sh = (0, 0, 0, 0, 0, 1) 
        elif name == 'uonvx':
            arr = un
            sh = (0, 0, -1, 0, 0, 0) 
        elif name == 'vonuy':
            arr = vn
            sh = (0, 1, 0, 0, 0, 1) 
        elif name == 'wonu':
            arr = wn
            sh = (0, 1, 0, 0, 0, 1)
        elif name == 'uonvx':
            arr = un
            sh = (0, 0, -1, 0, 0, 0) 
        elif name == 'vonux':
            arr = vn
            sh = (-1, 0, -1, 0, 0, 0) 
        elif name == 'wonv':
            arr = wn
            sh = (0, 0, -1, 0, 0, 0)
        elif name == 'honu':
            arr = hn
            sh = (0, 1, 0, 0, 0, 1)
        elif name == 'honv':
            arr = hn
            sh = (0, 0, -1, 0, 0, 0)
       
        fields[name] = vectorized_loop(mode_num, N_y, N_x, arr, sh, field)
        
    return fields

#------------------------------------------------------------------------------
    
def processing( o, mode_num, N_y, N_x, nlparts ):
    
    out = dict()
    for i in ('nl_u', 'nl_v', 'nl_h'):
        out[i] = [np.zeros((N_y, N_x))] * mode_num
    
    for name, val in nlparts.items():
        if name == 'a1' or name == 'a2' or name == 'a3':
            for m in range(mode_num):
                out['nl_u'][m][:, :] += val[m, :, :]
        elif name == 'b1' or name == 'b2' or name == 'b3':
            for m in range(mode_num):
                out['nl_v'][m][:, :] += val[m, :, :]
        elif name == 'c1' or name == 'c2' or name == 'c3' or name == 'c4':
            for m in range(mode_num):
                out['nl_h'][m][:, :] += val[m, :, :]
                
    #factoring 
    for name, val in out.items():
        if name == 'nl_u' or name == 'nl_v':
            for m in range(mode_num):
                out[name][m] *= (-1)
        elif name == 'nl_h':
            for m in range(mode_num):
                out[name][m] *= o.H[m] / o.rho_0
        else:
            print('Wrong naming!')
    
    return out
            
    
            
            
            
        
