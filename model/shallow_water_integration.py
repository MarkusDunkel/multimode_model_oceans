#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 09:58:16 2020

@author: markus
"""

import variables as var
import numpy as np 
from numba import jit
import ray

#==============================================================================
#-------------------------Beginn SWM Functions---------------------------------
#==============================================================================

@jit(nopython=True)
def zonal_pressure_gradient(state):
    """Compute zonal acceleration due to zonal pressure gradient."""
    u, v, eta, g, dx, halo = state
    u_t = np.zeros_like(u)
    
    # interior points
    for j in range(halo[0], u.shape[-2]-halo[0]):
        for i in range(halo[1], u.shape[-1]-halo[1]):
            u_t[j, i] = -g / dx * (eta[j, i] - eta[j, i-1])
  
    return u_t

#------------------------------------------------------------------------------

@jit(nopython=True)
def meridional_pressure_gradient(state):
    """Compute meridional acceleration due to zonal pressure gradient."""
    
    u, v, eta, g, dy, halo = state
    v_t = np.zeros_like(v)

    # interior points
    for j in range(halo[0], v.shape[-2] - halo[0]):
        for i in range(halo[1], v.shape[-1] - halo[1]):
            v_t[j, i] = -g / dy * (eta[j - 1, i] - eta[j, i])
            
    return v_t

#------------------------------------------------------------------------------

@jit(nopython=True)
def add_zonal_windstress(t, tau_x, wind_up, rho_0, p_null, halo, shape):
    """Compute acceleration by the wind stress in zonal direction"""
    
    if t <= wind_up:
        tau_x = tau_x / wind_up * t
    
    a = np.zeros(shape)
    for j in range(halo[0], a.shape[0]-halo[0]):
        for i in range(halo[1], a.shape[1]-halo[1]):
            a[j, i] = tau_x[j-halo[0], i-halo[1]] * 1/rho_0 * p_null
            
    return a

#------------------------------------------------------------------------------

@jit(nopython=True)
def add_meridional_windstress(t, tau_y, wind_up, rho_0, p_null, halo, shape):
    """Compute acceleration by the wind stress in meridional direction"""
    
    if t <= wind_up:
        tau_y = tau_y / wind_up * t
        
    a = np.zeros(shape)
    for j in range(halo[0], a.shape[0]-halo[0]):
        for i in range(halo[1], a.shape[1]-halo[1]):
            a[j, i] = tau_y[j-halo[0], i-halo[1]] * 1/rho_0 * p_null
    
    return a

#------------------------------------------------------------------------------

@jit(nopython=True)
def add_zonal_coriolis(state):
    u, v, eta, f_on_u, halo = state
    """Compute Coriolis acceleration in zonal direction"""
    
    u_t = np.zeros_like(v)
    for j in range(halo[0], u.shape[0]-halo[0]):
        for i in range(halo[1], u.shape[1]-halo[1]):
            u_t[j, i] = f_on_u[j-halo[0], i-halo[1]] /4 * (
                v[j, i-1] + v[j, i] + v[j+1, i] + v[j+1, i-1])
                                                          
    return u_t

#------------------------------------------------------------------------------

@jit(nopython=True)
def add_mer_coriolis(state):
    u, v, eta, f_on_v, halo = state
    """Compute Coriolis acceleration in meridional direction"""

    v_t = np.zeros_like(v)
    for j in range(halo[0], v_t.shape[0]-halo[0]):
        for i in range(halo[1], v_t.shape[1]-halo[1]):
            v_t[j, i] = f_on_v[j-halo[0], i-halo[1]] /4 * (
                u[j-1, i] + u[j-1, i+1] + u[j, i+1] + u[j, i])
                                                          
    return v_t

#------------------------------------------------------------------------------
    
@jit(nopython=True)
def ts_adams_bashforth_3(v_n, g_n, g_nm1, g_nm2, dt, halo):
    """Integrate using Adams-Bashforth 3 Level time stepping."""
    
    ny, nx = v_n.shape

    v_np1 = np.zeros_like(v_n)
    for j in range(halo[0], ny-halo[0]):
        for i in range(halo[1], nx-halo[1]):
            v_np1[j, i] = (
                v_n[j, i]
                + dt / 12. * (
                    23. * g_n[j, i] - 16. * g_nm1[j, i] + 5. * g_nm2[j, i]
                )
            )
    return v_np1

#------------------------------------------------------------------------------
    
@jit(nopython=True)
def ts_adams_bashforth_2(v_n, g_n, g_nm1, dt, halo):
    """Integrate using Adams-Bashforth 2 Level time stepping."""
    
    ny, nx = v_n.shape

    v_np1 = np.zeros_like(v_n)
    for j in range(halo[0], ny-halo[0]):
        for i in range(halo[1], nx-halo[1]):
            v_np1[j, i] = (
                v_n[j, i] + dt / 2. * (3. * g_n[j, i] - g_nm1[j, i]) )
            
    return v_np1

#------------------------------------------------------------------------------
    
@jit(nopython=True)
def ts_adams_bashforth_1(v_n, g_n, dt, halo):
    """Integrate using Adams-Bashforth 1 Level time stepping."""
    
    ny, nx = v_n.shape

    v_np1 = np.zeros_like(v_n)
    for j in range(halo[0], ny-halo[0]):
        for i in range(halo[1], nx-halo[1]):
            v_np1[j, i] = v_n[j, i] + dt * g_n[j, i]
            
    return v_np1

#------------------------------------------------------------------------------

@jit(nopython=True)
def add_zonal_visc(state):
    """Compute horizontal viscosity  in the zonal direction"""
    u, vis, dx, dy, halo, shape = state
    ux2_n = np.zeros(shape)
    uy2_n = np.zeros(shape)
    visc_x = np.zeros(shape)
    
    for j in range(halo[0], u.shape[0]-halo[0]):
        for i in range(halo[1], u.shape[1]-halo[1]):
            
            ux2_n[j, i] = (u[j, i-1] - 2*u[j, i] + u[j, i+1])/dx**2
            uy2_n[j, i] = (u[j-1, i] - 2*u[j, i] + u[j+1, i])/dy**2
             
            visc_x[j, i] = vis*(ux2_n[j, i] + uy2_n[j, i])
    
    return visc_x

#------------------------------------------------------------------------------

@jit(nopython=True)
def add_mer_visc(state):
    """Compute horizontal viscosity  in the meridional direction"""
    v, vis, dx, dy, halo, shape = state
    vx2_n = np.zeros(shape)
    vy2_n = np.zeros(shape)
    visc_y = np.zeros(shape)
    
    for j in range(halo[0], v.shape[0]-halo[0]):
        for i in range(halo[1], v.shape[1]-halo[1]):
            
            vx2_n[j, i] = (v[j, i-1] - 2*v[j, i] + v[j, i+1])/dx**2
            vy2_n[j, i] = (v[j-1, i] - 2*v[j, i] + v[j+1, i])/dy**2
             
            visc_y[j, i] = vis*(vx2_n[j, i] + vy2_n[j, i])
    
    return visc_y

#------------------------------------------------------------------------------

@jit(nopython=True)
def add_den_visc(state):
    """Compute horizontal diffusivity in density"""
    h, visd, dx, dy, halo, shape = state
    hx2_n = np.zeros(shape)
    hy2_n = np.zeros(shape)
    visc_h = np.zeros(shape)
    
    for j in range(halo[0], h.shape[0]-halo[0]):
        for i in range(halo[1], h.shape[1]-halo[1]):
            
            hx2_n[j, i] = (h[j, i-1] - 2*h[j, i] + h[j, i+1])/dx**2
            hy2_n[j, i] = (h[j-1, i] - 2*h[j, i] + h[j+1, i])/dy**2
             
            visc_h[j, i] = visd*(hx2_n[j, i] + hy2_n[j, i])
    
    return visc_h

#------------------------------------------------------------------------------
    
@jit(nopython=True)
def add_linfric(state):
    """Compute Rayleigh friction in momentum"""
    
    A, H, g, vel, halo = state
    vel_t = np.zeros_like(vel)
    
    for j in range(halo[0], vel.shape[0]-halo[0]):
        for i in range(halo[1], vel.shape[1]-halo[1]):
            vel_t[j, i] = A / (g * H) * vel[j, i]
            
    return vel_t
            
#==============================================================================
#----------------------------End SWM Functions---------------------------------
#==============================================================================

def swm_integration( t, o, m, shape, halo, f_on_u, f_on_v, u_n, v_n, h_n, w_n,
        gu_n, gv_n, gh_n, gu_nm1, gv_nm1, gh_nm1, gu_nm2, gv_nm2, gh_nm2):
    
    gu_n += zonal_pressure_gradient(
        (u_n, v_n, h_n, o.g, o.dx, o.halo))
    gv_n += meridional_pressure_gradient(
        (u_n, v_n, h_n, o.g, o.dy, o.halo))  
    
    # compute tendency due to Coriolis force
    if (o.use_coriolis is True):
        gu_n += add_zonal_coriolis(
            (u_n, v_n, h_n, f_on_u, halo))
        gv_n -= add_mer_coriolis(
            (u_n, v_n, h_n, f_on_v, halo))
     
    # compute tendency due to Eddy Viscosity
    if (o.use_eddy_visc is True):
        gu_n += add_zonal_visc(
            (u_n, o.vis, o.dx, o.dy, halo, shape))
        gv_n += add_mer_visc(
            (v_n, o.vis, o.dx, o.dy, halo, shape))
    if (o.use_eddy_diff is True):
        gh_n += add_den_visc(
            (h_n, o.visd, o.dx, o.dy, halo, shape))
        
    # compute tendency due to wind stress
    if (o.use_wind is True):
        gu_n += add_zonal_windstress(
            t, o.tau_x, o.wind_up_t, o.rho_0, o.pmode_null[m], halo, shape) 
        gv_n += add_meridional_windstress(
            t, o.tau_y, o.wind_up_t, o.rho_0, o.pmode_null[m], halo, shape)
        
    # linear friction
    if (o.use_linfric is True):
        gu_n -= add_linfric((o.A, o.H[m], o.g, u_n, halo))
        gv_n -= add_linfric((o.A, o.H[m], o.g, v_n, halo))
        gh_n -= add_linfric((o.B, o.H[m], o.g, h_n, halo))
        
    # compute tendency term of the continuity equation
    gh_n += w_n 
    
    # integrate in time
    if t > 1:
        u_np1 = ts_adams_bashforth_3(
            u_n, gu_n, gu_nm1, gu_nm2, o.dt, halo)
        v_np1 = ts_adams_bashforth_3(
            v_n, gv_n, gv_nm1, gv_nm2, o.dt, halo)
        h_np1 = ts_adams_bashforth_3(
            h_n, gh_n, gh_nm1, gh_nm2, o.dt, halo)
    elif t > 0:
        u_np1 = ts_adams_bashforth_2(
            u_n, gu_n, gu_nm1, o.dt, halo)
        v_np1 = ts_adams_bashforth_2(
            v_n, gv_n, gv_nm1, o.dt, halo)
        h_np1 = ts_adams_bashforth_2(
            h_n, gh_n, gh_nm1, o.dt, halo)
    elif t > -1:
        u_np1 = ts_adams_bashforth_1(
            u_n, gu_n, o.dt, halo)
        v_np1 = ts_adams_bashforth_1(
            v_n, gv_n, o.dt, halo)
        h_np1 = ts_adams_bashforth_1(
            h_n, gh_n, o.dt, halo)
    else: print('No appropriate time-stepping scheme was found.')
        
    # shift tendency terms
    gu_nm2 = gu_nm1.copy()
    gu_nm1 = gu_n.copy()   
    gv_nm2 = gv_nm1.copy() 
    gv_nm1 = gv_n.copy()  
    gh_nm2 = gh_nm1.copy() 
    gh_nm1 = gh_n.copy()   
    
    u_n = u_np1
    v_n = v_np1
    h_n = h_np1
    
    test = np.isnan(np.sum(h_n))
    if test:
        print('nan occured in fields; Model run became probably unstable.')
    
    state = { 'u': u_n, 'v': v_n, 'h': h_n, 'gu_nm1': gu_nm1, 'gv_nm1': gv_nm1, 
             'gh_nm1': gh_nm1, 'gu_nm2': gu_nm2, 'gv_nm2': gv_nm2, 
             'gh_nm2': gh_nm2 }
               
    return state
        