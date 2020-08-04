#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Domain module.

Created on Thu Jan 21 15:04:00 2020

@author: Markus Dunkel, adapted version from Prof. Martin Claus
"""
import numpy as np
import ray
import variables as var
import create_subdomain as cs


class Domain(object):
    """
    Domain class.
    Base domain class which may be used for global non-periodic domains. There
    are no halo points.
    """

    def __init__(self, o):
        halo = (0, 0)
        shape = (o.N_y, o.N_x, o.N_z)
        self.ndim = len(shape)
        self.shape = tuple(shape[:-1])
        self.shape_3d = shape
        self.vars = dict()
        self.vars_3d = dict()
        self._local_slice = self.ndim * (slice(None),)
        self.halo = halo
        self.double_periodic = o.double_periodic

    def create_var(self, mode_num, *args):
        """Create a variable on the domain.

        """
        for name in args:
            if name in self.vars:
                raise RuntimeError(
                    "Try to create variable {} that already exists!".format(
                        name)
                )
            self.vars[name] = var.TwoD( 
                np.zeros(self.shape), mode_num, self.halo )

    def get_var(self, name):
        """Return a view to the data of the variable without ghost points."""
        return self.vars[name][self._local_slice]
    
    def gather(self, handles, sub_slices):
        """Collect variable from subdomains into a global array.

        """
        
        for s in range(len(handles)):
            
            diclo = ray.get(
                handles[s].send_variables_to_global_domain.remote() )
            for key, value in diclo.items():
                self.vars[key].write_to_all_modes( 
                    diclo[key].get_all_modes(
                        diclo[key]._local_slice), sub_slices[s] )


@ray.remote
class SubDomain():
    """Subdomain class.
    A domain is decomposed into subdomains by a decomposer object/class.
    Each subdomain knows about its neighbours and how to retrieve information
    for the halo points from them.
    """

    def __init__(self, sid, main, o):
        """Construct a SubDomain object.

        """
        subdomains = cs.create_subdomains(o, sid, o.halo, o.domain_split, main)
        
        neighbours, boundaries, slip, fields, meta, shape = subdomains[0]
                 
        self.neighbours = neighbours[0]
        self.neighbour_to_self = neighbours[1]
        self.solid_wall_boundaries = boundaries
        self.slip_boundary_condition = slip
        self.vars = fields
        self.o = meta
        self.halo = o.halo
        self.shape = shape
        self.sid = sid
        self.tau_x = meta.tau_x
        self.tau_y = meta.tau_y
        self.f_on_u = meta.f_on_u
        self.f_on_v = meta.f_on_v
        
        
        if len(shape) != len(self.halo):
            raise RuntimeError(
                'The length of halo does not match the length of shape'
            )
       
        self._local_slice = tuple(
            slice(None) if h == 0 else slice(h, -h) for h in self.halo
        )

        
    def communication_between_subdomains(self, domain_handles, *args):
        """Get data from neighbours for a given variable.

        """
        
        for name in args:
        
            for i in range(len(self.neighbours)):
                self._get_from_neighbour(self.neighbours[i],
                            domain_handles[self.neighbours[i].domain], name) 
               
            if self.neighbour_to_self !=[]:
                self._get_from_self(self.neighbour_to_self, name)
            
        return None
    
    def get_slip_boundary_values(self):
        
        self._get_from_self(self.slip_boundary_condition[0], 'u')
        self._get_from_self(self.slip_boundary_condition[1], 'v')
        self._get_from_self(self.slip_boundary_condition[2], 'h')
        
    def apply_solid_wall_boundary_condition(self):
        
        self.vars['u'].receive_flatten_array( 
            0, self.solid_wall_boundaries[0] )
        self.vars['v'].receive_flatten_array( 
            0, self.solid_wall_boundaries[1] )
    
    def _get_from_self(self, neighbour_to_self, name):
        
        self.vars[name].receive_flatten_array(
            self.vars[name].get_flatten_array(
                self._local_slice, neighbour_to_self.remote_indx
                ), neighbour_to_self.local_indx ) 
        

    def _get_from_neighbour(self, neighbour, handle, name):
        
        local_buffer = ray.get( handle.send_data_out.remote(
            name, neighbour.remote_indx
        ) )
        
        self.vars[name].receive_flatten_array(
            local_buffer, neighbour.local_indx)
        
    def send_data_out(self, name, indx):
        """Return data requested by remote domain.

        """
        data = self.vars[name].get_flatten_array(
            self.vars[name]._local_slice, indx)
                                     
        return data 
    
    def get_variables(self, names = False):
        
        if names:
            dict_out = { i: self.vars[i] for i in names }
        else:
            dict_out = self.vars
        
        return dict_out
    
    def get_attributes(self, *args):
        out = []
        for arg in args:
            out.append( self.__dict__[arg] )
        return out
            
    def update_variables_via_dic(self, modes, state):
        for j in range(len(state)):
            variables = state[j]
            for key, value in variables.items():
                self.vars[key].wom(modes[j], value)
                
    def update_variables_via_TwoD_like(self, state):
        for key, value in state.items():
            self.vars[key].write_to_all_modes(value)
            
    def update_parts_of_modes(self, state): 
        for key, value in state.items():
            self.vars[key].add_to_all_modes(value)
         
    def send_class_copy_to_local_node( self ):
        return self
    
    def send_variables_to_global_domain(self):
        return self.vars
    
    
            
class Neighbour(object):
    """Simple class to store neighborhood information.
    
    """

    def __init__(self, dummy):
        domain, remote_indx, local_indx = dummy
        """Construct neighborhood information.
        
        """
        self.domain = domain
        self.remote_indx = remote_indx
        self.local_indx = local_indx

                
class Checkerboard2D():
    """Checkerboard domain decomposition along last two dimensions.
    
    """
    def __init__(self, global_domain, o):
        self.subdomains = []
        self.schedule = []
        self.nrow, self.ncol = o.domain_split
        self.sub_num = self.nrow * self.ncol
        self.sids = tuple(range(self.sub_num))
        self.sub_slices = []
        for sid in self.sids:
            self.subdomains.append( SubDomain.remote(sid, global_domain, o) )  
        self.sub_slices = cs.get_subdomain_slices(
            global_domain.shape, self.nrow, self.ncol) 
        self.schedule = cs.get_subdomain_schedule( o )
            
        
        
    