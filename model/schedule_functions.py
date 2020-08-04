#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:42:38 2020

@author: Markus Dunkel
"""

import numpy as np 
import ray 
import shallow_water_integration as swi
import vertical_mixing as ver
import non_linear as nlin
from numba import jit
import time
from itertools import combinations_with_replacement
from itertools import product

@ray.remote
def integrate_SWM(domain, t, o, bunch_of_swm_keys):
    v = ray.get( domain.get_variables.remote() )
    domain_attr = ray.get( domain.get_attributes.remote(
                                                 'shape', 'f_on_u', 'f_on_v') )
    shape = domain_attr[0]
    f_on_u = domain_attr[1]
    f_on_v = domain_attr[2]
    
    # adding up all external forcings
    gu_n = [None] * o.mode_num
    gv_n = [None] * o.mode_num
    gh_n = [None] * o.mode_num
    
    for m in bunch_of_swm_keys:
        gu_n[m] = v['vmix_u'].gom(m) + v['nl_u'].gom(m) 
        gv_n[m] = v['vmix_v'].gom(m) + v['nl_v'].gom(m) 
        gh_n[m] = v['vmix_h'].gom(m) + v['nl_h'].gom(m) 
    
    state = []
    for m in bunch_of_swm_keys: 
        state.append( swi.swm_integration( t, o, m, shape, o.halo, f_on_u, 
        f_on_v, v['u'].gom(m), v['v'].gom(m), v['h'].gom(m), v['w'].gom(m), 
        gu_n[m], gv_n[m], gh_n[m], v['gu_nm1'].gom(m), 
        v['gv_nm1'].gom(m), v['gh_nm1'].gom(m), v['gu_nm2'].gom(m), 
        v['gv_nm2'].gom(m), v['gh_nm2'].gom(m) ) )
    
    domain.update_variables_via_dic.remote( bunch_of_swm_keys, state )
    

@ray.remote
def apply_boundary_conditions(domain, o):
    
    if not o.double_periodic: 
    # set solid wall boundary condition
        domain.apply_solid_wall_boundary_condition.remote()
    
    if o.free_slip:
        domain.get_slip_boundary_values.remote()
        
@ray.remote
def compute_vertical_mixing(domain, o):
    start_mix = time.time()
    
    v = ray.get( domain.get_variables.remote() )
    domain_attr = ray.get( domain.get_attributes.remote(
                                                   'shape', 'tau_x', 'tau_y') )
    shape = domain_attr[0]
    tau_x = domain_attr[1]
    tau_y = domain_attr[2]
    
    
    nzz_u = ver.compute_mixing_momentum(
        (o, tau_x, tau_y, (shape[0], shape[1], o.N_z), o.halo, 
        v['u'], 'momentum_u') )
    
    nzz_v = ver.compute_mixing_momentum(
        (o, tau_x, tau_y, (shape[0], shape[1], o.N_z), o.halo, 
        v['v'], 'momentum_v') )
    
    nzz_h = ver.compute_mixing_density( 
        (o, (shape[0], shape[1], o.N_z), o.halo, v['h']) ) 
    
    
    state = {'vmix_u': nzz_u, 'vmix_v': nzz_v, 'vmix_h': nzz_h}
    domain.update_variables_via_TwoD_like.remote( state )
    
    end_mix = time.time()
    print('Mixing takes ' + str(end_mix-start_mix) + ' seconds.')
    
    
@ray.remote
def compute_vertical_mixing_u(domain, o):
    start_mix = time.time()
    
    v = ray.get( domain.get_variables.remote() )
    domain_attr = ray.get( domain.get_attributes.remote(
                                                   'shape', 'tau_x', 'tau_y') )
    shape = domain_attr[0]
    tau_x = domain_attr[1]
    tau_y = domain_attr[2]
    
    nzz_u = ver.compute_mixing_momentum(
        (o, tau_x, tau_y, (shape[0], shape[1], o.N_z), o.halo, 
        v['u'], 'momentum_u') )
     
    state = {'vmix_u': nzz_u}
    
    end_mix = time.time()
    print('Mixing u takes ' + str(end_mix-start_mix) + ' seconds.')
    
    domain.update_variables_via_TwoD_like.remote( state )
    
    
@ray.remote
def compute_vertical_mixing_v(domain, o):
    start_mix = time.time()
    
    v = ray.get( domain.get_variables.remote() )
    domain_attr = ray.get( domain.get_attributes.remote(
                                                   'shape', 'tau_x', 'tau_y') )
    shape = domain_attr[0]
    tau_x = domain_attr[1]
    tau_y = domain_attr[2]
    
    nzz_v = ver.compute_mixing_momentum(
        (o, tau_x, tau_y, (shape[0], shape[1], o.N_z), o.halo, 
        v['v'], 'momentum_v') )
    
    state = {'vmix_v': nzz_v}
    
    end_mix = time.time()
    print('Mixing v takes ' + str(end_mix-start_mix) + ' seconds.')
    
    domain.update_variables_via_TwoD_like.remote( state )
    
    
@ray.remote
def compute_vertical_mixing_h(domain, o):
    start_mix = time.time()
    
    v = ray.get( domain.get_variables.remote() )
    domain_attr = ray.get( domain.get_attributes.remote('shape') )
    shape = domain_attr[0]
    
    nzz_h = ver.compute_mixing_density( 
        (o, (shape[0], shape[1], o.N_z), o.halo, v['h']) ) 
    
    state = {'vmix_h': nzz_h}
    
    end_mix = time.time()
    print('Mixing h takes ' + str(end_mix-start_mix) + ' seconds.')
    
    domain.update_variables_via_TwoD_like.remote( state )
    
@jit(nopython=True)
def massflux_divergence(state):
    """Compute the divergence of the mass flux."""
    u_n, v_n, H, dx, dy, g, halo = state
    
    w_n = np.zeros_like(u_n)
    # interior points
    for j in range(halo[0], w_n.shape[-2] - halo[0]):
        for i in range(halo[1], w_n.shape[-1] - halo[1]):
            w_n[j, i] = (-1) * H * (
                (u_n[j, i+1] - u_n[j, i]) / dx + (v_n[j, i] - v_n[j+1, i]) / dy
                )
            
    return w_n

    
@ray.remote
def compute_w(domain, o):
    
    v = ray.get( domain.get_variables.remote() )
    domain_attr = ray.get( domain.get_attributes.remote('halo') )
    halo = domain_attr[0]
    
    w_n = []
    for m in range(o.mode_num):
        state = (v['u'].gom(m), v['v'].gom(m), o.H[m], o.dx, o.dy, o.g, halo)
        w_n.append( massflux_divergence(state) )
    
    state = {'w': w_n}
    domain.update_variables_via_TwoD_like.remote( state )
    

@ray.remote
def compute_nonlinear_free_split(domain, o, names):
    start_nlin = time.time()
    
    comb = np.array(list(combinations_with_replacement(range(o.mode_num), 2 ))) 
    # ordes does not matter
    per = np.array(list(product(range(o.mode_num), repeat=2 ))) 
    # order does matter
    
    v = ray.get( domain.get_variables.remote() )
   
    un = np.asarray(v['u'].data)
    vn = np.asarray(v['v'].data)
    wn = np.asarray(v['w'].data)
    hn = np.asarray(v['h'].data)
    mode_num, N_y, N_x = np.shape(un)

    shfields = dict()
    for term in names:
        if term == 'a1':
            shfields['uonvy'] = None
        elif term == 'a2':
            shfields['uonvx'] = None
            shfields['vonuy'] = None
        elif term == 'a3':
            shfields['wonu'] = None
        elif term == 'b1':
            shfields['uonvx'] = None
            shfields['vonuy'] = None
        elif term == 'b2':
            shfields['vonux'] = None 
        elif term == 'b3':
            shfields['wonv'] = None 
        elif term == 'c1':
            shfields['honu'] = None
        elif term == 'c2':
            shfields['honv'] = None
                    
    shfields = nlin.shift_fields( mode_num, N_y, N_x, un, vn, hn, wn, shfields)  
     
    nlparts = dict()
    for term in names:
        if term == 'a1':
            arr = nlin.uu_x( 
                o.mode_num, N_y, N_x, o.dx, o.halo, shfields['uonvy'], comb )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.PPP, arr )
            #print(arr)
        elif term == 'a2':
            arr = nlin.uv_y( 
                o.mode_num, N_y, N_x, o.dx, o.halo, 
                shfields['uonvx'], shfields['vonuy'], per )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.PPP, arr )
        elif term == 'a3':
            arr = nlin.uw_or_vw( 
                o.mode_num, N_y, N_x, o.halo, un, shfields['wonu'], per )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.PPWz, arr )
        elif term == 'b1':
            arr = nlin.uv_x( 
                o.mode_num, N_y, N_x, o.dx, o.halo, 
                shfields['uonvx'], shfields['vonuy'], per )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.PPP, arr )
        elif term == 'b2':
            arr = nlin.vv_y( 
                o.mode_num, N_y, N_x, o.dy, o.halo, shfields['vonux'], comb )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.PPP, arr ) 
        elif term == 'b3':
            arr = nlin.uw_or_vw( 
                o.mode_num, N_y, N_x, o.halo, vn, shfields['wonv'], per )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.PPWz, arr )
        elif term == 'c1':
            arr = nlin.uh_x( 
                o.mode_num, N_y, N_x, o.dx, o.halo, un, shfields['honu'], per )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.WPWdz, arr )
        elif term == 'c2':
            arr = nlin.vh_y( 
                o.mode_num, N_y, N_x, o.dy, o.halo, vn, shfields['honv'], per )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.WPWdz, arr )
        elif term == 'c3':
            arr = nlin.uw_or_vw( o.mode_num, N_y, N_x, o.halo, hn, wn, per )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.WWWzdz, arr )
        elif term == 'c4':
            arr = nlin.uw_or_vw( o.mode_num, N_y, N_x, o.halo, hn, wn, per )
            nlparts[term] = nlin.vertical_expansion( 
                o.mode_num, N_y, N_x, o.WWWdzz, arr )
               
    out = nlin.processing( o, mode_num, N_y, N_x, nlparts )
    domain.update_parts_of_modes.remote( out )
    end_nlin = time.time()
    
    print('Non-linear takes ' + str(end_nlin-start_nlin) + ' seconds.')


@ray.remote
def compute_nonlinear_u(domain, o):
    start_nlin = time.time()
    
    v = ray.get( domain.get_variables.remote() )
    
    F_u = nlin.compute_nonlinear_u(o, v['u'].data, v['v'].data, v['w'].data)
    
    state = {'nl_u': F_u}
    domain.update_variables_via_TwoD_like.remote( state )
    
    end_nlin = time.time()
    print('Non-linear u takes ' + str(end_nlin-start_nlin) + ' seconds.')
    
@ray.remote
def compute_nonlinear_v(domain, o):
    start_nlin = time.time()
    
    v = ray.get( domain.get_variables.remote() )
    
    F_v = nlin.compute_nonlinear_v(o, v['u'].data, v['v'].data, v['w'].data)
    
    state = {'nl_v': F_v}
    domain.update_variables_via_TwoD_like.remote( state )
    
    end_nlin = time.time()
    print('Non-linear v takes ' + str(end_nlin-start_nlin) + ' seconds.')
    
@ray.remote
def compute_nonlinear_h(domain, o):
    start_nlin = time.time()
    
    v = ray.get( domain.get_variables.remote() )
    
    F_h = nlin.compute_nonlinear_h(
        o, v['u'].data, v['v'].data, v['h'].data, v['w'].data)
    
    state = {'nl_h': F_h}
    domain.update_variables_via_TwoD_like.remote( state )
    
    end_nlin = time.time()
    print('Non-linear h takes ' + str(end_nlin-start_nlin) + ' seconds.')
    
    
