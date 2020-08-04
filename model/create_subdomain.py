#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:13:19 2020

@author: Markus Dunkel; adapted version from Prof. Martin Claus;
"""

import numpy as np 
import copy
import domain as dom
import variables as var


def _get_boundaries( sid, nrow, ncol, halo, shape, pd):
    
    mask = np.full(shape, True)
    if not pd:
        if sid < ncol:
            mask[:halo[0], :] = False
        if sid >= (ncol * (nrow-1)):
            mask[-halo[0]:, :] = False
        if sid in range(ncol-1, ncol*nrow, ncol):
            mask[:,-halo[1]:] = False
        if sid in range(0, ncol * nrow, ncol):
            mask[:, :halo[1]] = False
    
    return mask


def _get_full_shape(shape, halo):
    return tuple([s + 2 * h for s, h in zip(shape, halo)])


def _mod_index(index, shape):
    """Map index cyclicly to valid index space."""
    return tuple(
        (i % s + s) % s for i, s in zip(index, shape)
    )


def _flatten_index(shp):
    """Return flatten index for array with shape shp.

    Returned index array also has shape shp.
    """
    return np.arange(np.prod(shp), dtype=np.intp).reshape(shp)


def _slice_to_halo_slice(local_slice, dimlen, n_halo):
    """Extend slice to include halo rim."""
    start, stop, _ = local_slice.indices(dimlen)
    return slice(start-n_halo, stop+n_halo)


def _index_array_with_halo(slices, shape, halo):
    """Return index array with halo points.

    The shape of the return array is
    (ndim,) + _get_full_shape(shape, halo).
    Indices are with respect to an array of shape local_shape and stored along
    the first dimension.
    """
    return np.mgrid[
        tuple(
            map(
                _slice_to_halo_slice,
                slices, shape, halo
            )
        )
    ]


def _cyclic_index_array(local_slices, shape, halo):
    """Create a index array from a tuple of slice objects.

    The slice objects will be extended to include the halo rim defined by halo.
    The indices will be periodic for an array that has a shape as shape.
    """
    return np.apply_along_axis(
        _mod_index,
        axis=0,
        arr=_index_array_with_halo(local_slices, shape, halo),
        shape=shape,
    )


def _get_with_index_array(arr, index_array):
    
    return arr[tuple(index_array)]


def _get_subdomain_dim_length(n_glob, n_sub):
    length = np.array(n_sub * (n_glob // n_sub,))
    length[:n_glob-length.sum()] += 1
    return tuple(length)

#------------------------------------------------------------------------------

def create_subdomains(o, sids, halo, domain_split, main ):
    """Create a set of subdomains.

    There will be nrow times ncol subdomains.

    Parameters
    ----------
    halo : tupel of int
        number of halo points for each dimensions.
    nrow : int
        number of subdomains along second last dimension
    ncol : int
        Number of subdomains along last dimension.

    """
    if isinstance(sids, int):
        sids = [sids]
    
    nrow, ncol = domain_split
    main_shape = (o.N_y, o.N_x)
    nsub = nrow * ncol
    sids_full_list = tuple(range(nsub))

    # get slice tuple for subdomains
    sub_slices = get_subdomain_slices(main_shape, nrow, ncol)
    # create global arrays of subdomain IDs
    # and local index space for subdomanins
    sub_ids = np.empty((o.N_y, o.N_x), dtype=np.intp)
    local_index = np.empty_like(sub_ids)
    for s_id in sids_full_list:
        sub_slice = sub_slices[s_id]
        sub_ids[sub_slice] = s_id
        local_index[sub_slice] = _flatten_index(
            local_index[sub_slice].shape
        )

    # create subdomains
    shape_list = []
    for sid in sids:
        sub_slice = sub_slices[sid]
        local_shape = local_index[sub_slice].shape
        full_shape = _get_full_shape(local_shape, o.halo)
        shape_list.append( full_shape )
        

    # figure out the neighbour relations for each subdomain
    neighbours_list = []
    for i in range(len(sids)):
        sid = sids[i]
        sub_slice = sub_slices[sid]
        neighbours, neighbour_to_self = get_neighbours(
            main_shape, sid, halo, nrow, ncol, sub_slice, sub_ids, local_index, 
            o.edge_com, o.double_periodic )
        dummy=[]
        if neighbour_to_self != []:
            dummy = dom.Neighbour(neighbour_to_self)
            
        neighbours_list.append((tuple(map(dom.Neighbour, neighbours)), dummy ))
        
        
    # import fields as they are stored in the Domain class 
    fields_list = []
    for i in range(len(sids)):
        sid = sids[i]
        sub_vars = {}
        for key in main.vars:
            sub_vars[key] = var.TwoD(np.zeros(shape_list[i]), o.mode_num, halo)
            sub_vars[key].write_to_all_modes( 
                main.vars[key].get_all_modes( 
                    sub_slices[ sid ])
                , sub_vars[key]._local_slice)
        fields_list.append( sub_vars )
             
       
    # prepare data exchange for wall boundary condition
    boundaries_list = []
    for sid in sids:
        sub_slice = sub_slices[sid]
        boundary_u, boundary_v = get_wall_boundary_conditions( 
            main_shape, sid, halo, sub_slice, nrow, ncol)
        
        boundaries_list.append( (boundary_u, boundary_v) )
        
    
    # prepare data exchange for slip boundary condition. 
    slip_list = []
    for sid in sids:
        sub_slice = sub_slices[sid]
        local_shape = local_index[sub_slice].shape
        full_shape = _get_full_shape(local_shape, o.halo)
        slip_u, slip_v, slip_h = get_slip_boundary_conditions( main_shape, sid, 
                                    halo, sub_slice, local_index, 
                                    full_shape, nrow, ncol, o.double_periodic)
        
        slip_u = dom.Neighbour(slip_u)
        slip_v = dom.Neighbour(slip_v)
        slip_h = dom.Neighbour(slip_h)
        
        slip_list.append( (slip_u, slip_v, slip_h) )
     
    meta_list = []
    for sid in sids:
        sub_slice = sub_slices[sid]
        meta_list.append( meta_to_subdomain(o, sub_slice, sid) )
        
    subdomains = []    
    for i in range(len(sids)):
        subdomains.append( ( neighbours_list[i], 
                   boundaries_list[i], slip_list[i], fields_list[i],
                   meta_list[i], shape_list[i]) )
        
    return subdomains
        
#------------------------------------------------------------------------------
    
def get_wall_boundary_conditions( 
        main_shape, sid, halo, sub_slice, nrow, ncol ):
    """ To impose solid wall boundary conditions the first row of v 
    and the first column of u is set to zero (when halo points are excluded).
    """
    
    full_global_index = _cyclic_index_array(
        sub_slice,
        main_shape,
        halo
    )

    full_sub_local_index = _flatten_index(full_global_index.shape[1:])
    
    
    if sid in range(0, ncol * nrow, ncol):
        uwall_mask = np.full(full_sub_local_index.shape, False)
        uwall_mask[halo[0]:-halo[0], halo[1]] = True
        uwall_indx = full_sub_local_index[uwall_mask].ravel()
    else: uwall_indx = []
    
    if sid < ncol:
        vwall_mask = np.full(full_sub_local_index.shape, False)
        vwall_mask[halo[0],halo[1]:-halo[1]] = True
        vwall_indx = full_sub_local_index[vwall_mask].ravel()
    else: vwall_indx = []
    
    return uwall_indx, vwall_indx


def get_slip_boundary_conditions(
        main_shape, sid, halo, sub_slice, local_index, full_shape, nrow, ncol, 
        double_periodic):
    
    full_global_index = _cyclic_index_array(
        sub_slice,
        main_shape,
        halo
    )

    full_sub_local_index = _flatten_index(full_global_index.shape[1:])
    overlap_index = _get_with_index_array(local_index, full_global_index)
    shape = np.shape(full_sub_local_index)
    
    main_edge_mask = _get_boundaries( 
       sid, nrow, ncol, halo, shape, 
       double_periodic )
    
    uslip_target_mask = np.full( full_shape, False ) 
    for i in range(full_shape[0]):
        if np.sum(main_edge_mask[i,:]) == 0:
            if i < halo[0]:
                uslip_target_mask[i, halo[1]:-halo[1]] = True
            elif i >= (full_shape[0] - halo[0]):
                uslip_target_mask[i, halo[1]:-halo[1]] = True
    
    vslip_target_mask = np.full( full_shape, False ) 
    for j in range(full_shape[1]):
        if np.sum(main_edge_mask[:,j]) == 0:
            if j < halo[1]:
                vslip_target_mask[halo[0]:-halo[0], j] = True
            elif j >= (full_shape[1] - halo[1]):
                vslip_target_mask[halo[0]:-halo[0], j] = True
                
    hslip_target_mask = np.full( full_shape, False ) 
    for i in range(full_shape[0]):
        for j in range(full_shape[1]):
            if vslip_target_mask[i, j] or uslip_target_mask[i, j]:
                hslip_target_mask[i, j] = True
   
    u_overlap_index = overlap_index.copy()
    h_overlap_index = overlap_index.copy()
    v_overlap_index = overlap_index.copy()
    
    for j in range(full_shape[1]):
        u_overlap_index[slice(0,halo[0]), j] = overlap_index[halo[0],j]
        u_overlap_index[slice(-1, -halo[0]-2, -1), j] = \
            overlap_index[-halo[0]-1, j] 
            
        h_overlap_index[slice(0,halo[0]), j] = overlap_index[halo[0],j]
        h_overlap_index[slice(-1, -halo[0]-2, -1), j] = \
            overlap_index[-halo[0]-1, j] 
    for i in range(full_shape[0]):
        v_overlap_index[i, slice(0, halo[1])] = overlap_index[i, halo[1]]
        v_overlap_index[i, slice(-1, -halo[1]-2, -1)] = \
        overlap_index[i, -halo[1]-1]
        
        h_overlap_index[i, slice(0, halo[1])] = overlap_index[i, halo[1]]
        h_overlap_index[i, slice(-1, -halo[1]-1, -1)] = \
        overlap_index[i, -halo[1]-1]
      
    uslip_target_indx = full_sub_local_index[uslip_target_mask].ravel()
    vslip_target_indx = full_sub_local_index[vslip_target_mask].ravel()
    hslip_target_indx = full_sub_local_index[hslip_target_mask].ravel()
    uslip_source_indx = u_overlap_index[uslip_target_mask].ravel()
    vslip_source_indx = v_overlap_index[vslip_target_mask].ravel()
    hslip_source_indx = h_overlap_index[hslip_target_mask].ravel()
    
    uslip = ( sid, uslip_source_indx, uslip_target_indx )
    vslip = ( sid, vslip_source_indx, vslip_target_indx )
    hslip = ( sid, hslip_source_indx, hslip_target_indx )
                
    return uslip, vslip, hslip
    

def get_neighbours(
    main_shape, sid, halo, nrow, ncol, sub_slice, subdomain_ids, local_index, 
    edge_com, double_periodic ):
    """finds neighbour relations"""
    full_global_index = _cyclic_index_array(
        sub_slice,
        main_shape,
        halo
    )

    full_sub_local_index = _flatten_index(full_global_index.shape[1:])
    overlap_sids = _get_with_index_array(subdomain_ids, full_global_index)
    overlap_index = _get_with_index_array(local_index, full_global_index)
    
    halopoint_mask = np.full(np.shape(overlap_sids), True)
    halopoint_mask[halo[0]:-halo[0], halo[1]:-halo[1]] = False
    main_edge_mask = _get_boundaries( 
        sid, nrow, ncol, halo, np.shape(overlap_sids), 
        double_periodic )
    
    if not edge_com:
        halopoint_mask[:halo[0], :halo[1]] = False
        halopoint_mask[:halo[0], -halo[1]:] = False
        halopoint_mask[-halo[0]:, -halo[1]:] = False
        halopoint_mask[-halo[0]:, :halo[1]] = False
        
    neighbour_sids = np.unique(overlap_sids[halopoint_mask])
    
    neighbours = []

    neighbour_to_self = []
    for neighbour_sid in neighbour_sids:
        overlap = (overlap_sids == neighbour_sid)
        overlap = overlap * halopoint_mask * main_edge_mask
        
        source_index = overlap_index[overlap]
        target_index = full_sub_local_index[overlap].ravel()
        
        if neighbour_sid == sid:
            neighbour_to_self = (neighbour_sid, source_index, target_index)
        else:
            neighbours.append(
                (neighbour_sid, source_index, target_index) )
       
    return tuple(neighbours), neighbour_to_self


def get_subdomain_slices(main_shape, nrow, ncol):
    Ny, Nx = main_shape
    ndim = len(main_shape)#+1
    widths = _get_subdomain_dim_length(Nx, ncol)
    heights = _get_subdomain_dim_length(Ny, nrow)
    sub_slices = []
    for j in range(nrow):
        j0 = sum(heights[:j])
        j1 = j0 + heights[j]
        for i in range(ncol):
            i0 = sum(widths[:i])
            i1 = i0 + widths[i]
            sub_slices.append(
                (ndim - 2) * (slice(None),)
                + (slice(j0, j1), slice(i0, i1))
            )
    return tuple(sub_slices)


def meta_to_subdomain(o, subslice, sid):
    """Storing a modified copy of metadata dictionary on each subdomain"""
    dummy = copy.deepcopy( o )
    for key, val in dummy.__dict__.items():
        
        try:
            shape = np.shape( val )
        except: shape = ( 0, 0 )
 
        if shape == ( dummy.N_y, dummy.N_x ):
            dummy.__dict__[key] = val[subslice] 
            
    dummy.N_y = None 
    dummy.N_x = None
    dummy.y = None
    dummy.x = None
    dummy.id = sid
    return dummy


# Communication between subdomains need to be scheduled to avoid deadlock. 
def get_subdomain_schedule(o):
    
    schedule = []
    row = o.domain_split[0]
    col = o.domain_split[1]
    if not o.double_periodic:
        a = np.ones((row,col))
        set_back_arr = [0, 3, 1, 4, 2]
        rel = [2, 4, 3, 1]
        pos = 0
        count=0
        for i in range(row):
        
            pos = set_back_arr[count]
            count += 1
            if count > 4:
                count=0
            a[i, pos::5] = 0
            
            for j in range(4):
                os = pos - rel[j]
                if os < 0:
                    os += 5
                a[i, os::5] = j+1
            
        arr = a.reshape((-1,))
        
        for i in range(5):
            dummy = np.where(arr == i)
            schedule.append( dummy[0] )
        
        schedule = list(filter(lambda i: i.size != 0 , schedule))
            
    else:
        print('No schedule strategy for optimal subdomain \
              communication implemented yet.')
        
        for i in range(row * col):
            schedule.append([i])
            
    return schedule
            