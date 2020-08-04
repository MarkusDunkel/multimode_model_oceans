#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 19:09:22 2020

@author: Markus Dunkel
"""

import ray 
import init
import domain as do
import makefile as mf
import schedule_functions as mfunc
import time
import numpy as np
import variables as var

@ray.remote 
class Processing():
    """
    This class initate parallel processing of the output data like saving it in 
    an output file.  
    """
    def __init__(self, o, sub_slices):
        self.out = mf.OutputCompilation( o )
        self.sub_handles = []
        self.sub_slices = sub_slices
        self.mode_num = o.mode_num
        self.shape = (o.N_y, o.N_x, o.N_z)
        self.av_window = o.av_window
        self.zon_sect = o.zon_sect
        self.mer_sect = o.mer_sect
        self.variable_names = o.variable_names
        self.vars = dict()
        
        
        for name in self.variable_names:
            self.vars[name] = var.TwoD( 
                    np.zeros((self.shape[0], self.shape[1])), 
                    self.mode_num, (0,0) )
        
        
    def gather(self, handles, sub_slices):
       """Collect variable from subdomains into a global array."""
       
       for s in range(len(handles)):
           
           diclo = ray.get(handles[s].send_variables_to_global_domain.remote())
           for key, value in diclo.items():
               if key in self.variable_names:
                   self.vars[key].add_to_all_modes( 
                       diclo[key].get_all_modes(
                           diclo[key]._local_slice), sub_slices[s] )
               
  
    def process_data(self, o, t, domain_handles):
        """
        main function to process the output.
        """
        out_start = time.time()
        self.gather( domain_handles, self.sub_slices )
        
        if t in o.av_steps:
            for key, val in self.vars.items():
                for i in range(self.mode_num):
                    val.data[i] /= self.av_window
                    
            names = []
            for name in self.out.file_names:
                if name in self.out.continous_rep:
                    names.append(name)
                    
                if t in o.save_time_steps: 
                    if name in self.out.sample_rep:
                        names.append(name)
        
            self.out.write_to_ncfile(t, self.vars, names)
            
            for val in self.vars.values():
               for i in range(self.mode_num):
                   val.data[i][:] = 0
            
        out_end = time.time()
        print('Step ' + str(t) + ': Writing to output took ' + 
            str(out_end-out_start) + ' seconds.')
        
    def send_class_copy_to_local_node(self):
        """
        send a copy of the class to the global scope. 
        """
        return self


ray.shutdown()
ray.init()


o = init.MetaData()
o_id = ray.put( o )

main = do.Domain(o)

#initialization of all fields
main.create_var( o.mode_num, 'u', 'v', 'h', 'w', 'vmix_u', 'vmix_v', 'vmix_h', 
                'nl_u', 'nl_v', 'nl_h',
                'gu_nm1', 'gv_nm1', 'gh_nm1', 'gu_nm2', 'gv_nm2', 'gh_nm2' )


#Simple way to check if domain communication is done right. 
#Not part of the model. 
number_grid = False

if number_grid:
    k=1
    for i in range(o.N_y):
        for j in range(o.N_x):
            main.vars['u'].write_to_all_modes( 
                np.array([k]), (slice(i, i+1), slice(j, j+1)) )
            main.vars['v'].write_to_all_modes( 
                np.array([k]), (slice(i, i+1), slice(j, j+1)) )
            main.vars['h'].write_to_all_modes( 
                np.array([k]), (slice(i, i+1), slice(j, j+1)) )
            k+=1

sub = do.Checkerboard2D( main, o )
domain_handles = sub.subdomains
sub.subdomains = None

# stateful worker for processing the model output. 
if o.save_output: 
    output_handle = Processing.remote( o, sub.sub_slices )
    processing_ids = []

#==============================================================================
#------------------------- Start Model Integration ----------------------------
#==============================================================================
obj1_ids = []
proc_id = []
start_all = time.time() 
for t in range(o.max_time_step):
    
    end_all = time.time()
    print('step ' + str(t-1) + ' took ' + str(end_all-start_all) + ' seconds.')
    print(t)
    start_all = time.time()
    
    obj2_ids = []
    for s in range(sub.sub_num):
        obj2_ids.append( mfunc.compute_w.remote(domain_handles[s], o_id) )
    
    #--------------------------------------------------------------------------
    # Computation of non-linear forcing 
        
    if (o.use_non_linear is True):
        
        if o.nonlinear_split == 'equation_split':
            for s in range(sub.sub_num):
                obj2_ids.append( 
                    mfunc.compute_nonlinear_u.remote( 
                        domain_handles[s], o_id ) )
                obj2_ids.append( 
                    mfunc.compute_nonlinear_v.remote( 
                        domain_handles[s], o_id ) )
                obj2_ids.append( 
                    mfunc.compute_nonlinear_h.remote( 
                        domain_handles[s], o_id ) )
            
        elif o.nonlinear_split == 'free_split':
            obj3_ids = []
            for s in range(sub.sub_num):
                obj3_ids.append( 
                    domain_handles[s].update_variables_via_TwoD_like.remote(
                    {'nl_u': 0, 'nl_v': 0, 'nl_h': 0} ) )
                ray.wait(obj3_ids, num_returns = len(obj3_ids))
                for name in o.term_division.keys():
                    obj2_ids.append( 
                    mfunc.compute_nonlinear_free_split.remote( 
                        domain_handles[s], o_id, o.term_division[name] ) )        
      
    ray.wait(obj2_ids, num_returns = len(obj2_ids))
    ray.wait(obj1_ids, num_returns = len(obj1_ids))
    
    #--------------------------------------------------------------------------
    #SWM integration
    
    start_swm = time.time()
    obj_ids = []
    # model state of each subdomain is pushed forward 
    for s in range(sub.sub_num):
        for i in range(o.swm_split):
            bunch_of_swm = o.swm_keys[i]
            obj_ids.append( 
                mfunc.integrate_SWM.remote( 
                    domain_handles[s], t, o_id, bunch_of_swm ) )
    ray.wait(obj_ids, num_returns = len(obj_ids))
    end_swm = time.time()
    print('SWM integration takes ' + str(end_swm-start_swm) + ' seconds.')
    
    #-------------------------------------------------------------------------- 
    # Computation of vertical mixing with constant eddy viscosity.
    
    if (o.use_uniform_mixing is True):
       obj1_ids = []
       if o.mixing_split:
           for s in range(sub.sub_num):
               obj1_ids.append( 
                   mfunc.compute_vertical_mixing_h.remote( 
                       domain_handles[s], o_id ) )
               obj1_ids.append( 
                   mfunc.compute_vertical_mixing_u.remote( 
                       domain_handles[s], o_id ) )
               obj1_ids.append( 
                   mfunc.compute_vertical_mixing_v.remote( 
                       domain_handles[s], o_id ) )
          
       else:
           for s in range(sub.sub_num):
               obj1_ids.append( 
                   mfunc.compute_vertical_mixing.remote( 
                       domain_handles[s], o_id ) )        
      
    #--------------------------------------------------------------------------
    # Applying boundary conditions and subdomain communication.
               
    obj_ids = []
    for s in range(sub.sub_num):
        obj_ids.append( 
            mfunc.apply_boundary_conditions.remote( 
                domain_handles[s], o_id ) )
    ray.wait(obj_ids, num_returns = len(obj_ids))
    
    start_com = time.time()
    # Communication between subdomains. Communication for each subdomain is 
    #initated in the order as defined in 'sub.schedule'
    for c in range(len(sub.schedule)):
        obj_ids = []
        for s in sub.schedule[c]:
            obj_ids.append( 
                domain_handles[s].communication_between_subdomains.remote( 
                    domain_handles, 'u','v','h' ) ) 
        ray.wait(obj_ids, num_returns = len(obj_ids)) 
    
    end_com = time.time()
    print('Communication takes ' + str(end_com-start_com) + ' seconds.')
    ray.wait(obj_ids, num_returns = len(obj_ids)) 

    #--------------------------------------------------------------------------
    # processing of the model output
    
    if o.save_output:
        ray.wait(proc_id, num_returns = len(proc_id))
        proc_id = []
        proc_id.append( 
            output_handle.process_data.remote(o_id, t, domain_handles) )
    
#==============================================================================
#------------------------- End Model Integration ------------------------------
#============================================================================== 
        
subrun_local_copies=[]
for i in range(sub.sub_num):
    subrun_local_copies.append( ray.get(
        domain_handles[i].send_class_copy_to_local_node.remote()
        ) )
    
main.gather( domain_handles, sub.sub_slices ) 
 
if o.save_output:   
    process_copy= ray.get(output_handle.send_class_copy_to_local_node.remote()) 
    
  
ray.shutdown()
