#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 08:51:08 2020

@author: markus
"""
import numpy as np

def list_scenarios():
    scenarios = {
        'LW': {
            # ----------------------manage output------------------------------
            'save_as' : 'LW',                      # name of output files
            'save_output' : True,                  # saving output (T/F)
            'sample_number' : 100,                 # evenly spaced samples
            'av_window' : 58,                      # output average window 
            'zon_sect' : (45, 50, 55),             # loc of zonal sections
            'mer_sect' : (50, 100, 150),           # loc of meridional sect
            'variable_names' : ('u', 'v', 'h', 'w', 'vmix_u', 'vmix_v', 'vmix_h', 
                'nl_u', 'nl_v', 'nl_h'),           # variables to be safed
            # -----------parameters for numerical discretization---------------
            'max_time_step' : 20000,          # number timesteps
            'dt' : 1500,                      # timestep length [s]
            'N_x' : 200,                      # number zonal gridpoints
            'N_y' : 101,                      # number merid. gridpoints
            'N_z' : 200,                      # number vertical gridpoints 
            'L_x' : 5e+6,                     # zonal length of domain [m]
            'L_y' : 2.5e+6,                   # merid. length of domain [m]
            
            # -------------------------run model parallel----------------------
            'domain_split' : (1, 1),          # number of subdomains 
            'nonlinear_split' : 'free_split', # 'equation_split' / 'free_split'
            'term_division' : {},             # for free-split only
            'swm_split' : 3,                  # number processes SWMint
            'mixing_split' : True,            # strategy vert. mixing
            
            # ------------------boundary conditions----------------------------
            'edge_com' : False,          # communication corner neighbours
            'double_periodic' : False,   # double_periodic boundary condition
            'free_slip' : True,          # free-slip boundary condition
            'halo' : (1, 1),             # number of halo points each side
            
           
            # -------------------physical parameters--------------------------- 
            'mode_list' : np.arange(0, 25, 1),   # list involved modes
            'rho_0' :  1027,                    # reference density [kg/m^3]
            'g' : 10,                           # gravity [m/s^2]
            'phi' : 0,                          # central latitude of domain 
            'omega' : 7.2921150e-5,             # earths rotation [1/s]
            'a' : 6371e+3,                      # radius of earth
            'vis' : 2000,                       # horizontal visc [m^2/s]
            'tau_x_coef' : -0.05,               # const. windstress zonal 
            'tau_y_coef' : 0,                   # const. windstress mer.
            'wind_up_t' : 300,
            
            # only used for analytical version of vertical mixing
            'A' : 1.33e-7,                      # const. vert. mix. momentum 
            'B' : 1.33e-7,                      # const. vert. mix density 
            
            # only used for full version of vertical mixing 
            'nu' : 0.001,                       # const. vert. mix. momentum 
            'kap' : 0.0001,                     # const. vert. mix. density
            
            'use_wind' : True,            # windstress analytical version
            'use_coriolis' : True,        # equatorial beta-plane
            'use_eddy_visc' : True,       # horizonta viscosity
            'use_eddy_diff' : False,      # horizonta viscosity in density
            'use_linfric' : True,         # linear friction
            'use_uniform_mixing' : False, # full version vert. mix.
            'use_non_linear' : False,     # non linear terms
            
        },
        
        #======================================================================
        
        'LE': {
            # ----------------------manage output------------------------------
            'save_as' : 'LE',                # name of output files
            'save_output' : True,                 # saving output (T/F)
            'sample_number' : 100,                # evenly spaced samples
            'av_window' : 58,                     # output average window 
            'zon_sect' : (45, 50, 55),            # loc of zonal sections
            'mer_sect' : (50, 100, 150),          # loc of meridional sect
            'variable_names' : ('u', 'v', 'h', 'w', 'vmix_u', 'vmix_v', 'vmix_h', 
                'nl_u', 'nl_v', 'nl_h'),          # variables to be safed
            # -----------parameters for numerical discretization---------------
            'max_time_step' : 20000,          # number timesteps
            'dt' : 1500,                      # timestep length [s]
            'N_x' : 200,                      # number zonal gridpoints
            'N_y' : 101,                      # number merid. gridpoints
            'N_z' : 900,                      # number vertical gridpoints 
            'L_x' : 5e+6,                     # zonal length of domain [m]
            'L_y' : 2.5e+6,                   # merid. length of domain [m]
            
            # -------------------------run model parallel----------------------
            'domain_split' : (1, 1),          # number of subdomains 
            'nonlinear_split' : 'free_split', # 'equation_split' / 'free_split'
            'term_division' : {},             # for free-split only
            'swm_split' : 3,                  # number processes SWMint
            'mixing_split' : True,            # strategy vert. mixing
            
            # ------------------boundary conditions----------------------------
            'edge_com' : False,          # communication corner neighbours
            'double_periodic' : False,   # double_periodic boundary condition
            'free_slip' : True,          # free-slip boundary condition
            'halo' : (1, 1),             # number of halo points each side
            
           
            # -------------------physical parameters--------------------------- 
            'mode_list' : np.arange(0, 25, 1),  # list involved modes
            'rho_0' :  1027,                    # reference density [kg/m^3]
            'g' : 10,                           # gravity [m/s^2]
            'phi' : 0,                          # central latitude of domain 
            'omega' : 7.2921150e-5,             # earths rotation [1/s]
            'a' : 6371e+3,                      # radius of earth
            'vis' : 2000,                       # horizontal visc [m^2/s]
            'tau_x_coef' : 0.05,                # const. windstress zonal 
            'tau_y_coef' : 0,                   # const. windstress mer.
            'wind_up_t' : 300,
            
            # only used for analytical version of vertical mixing
            'A' : 1.33e-7,                      # const. vert. mix. momentum 
            'B' : 1.33e-7,                      # const. vert. mix density 
            
            # only used for full version of vertical mixing 
            'nu' : 0.001,                       # const. vert. mix. momentum 
            'kap' : 0.0001,                     # const. vert. mix. density
            
            'use_wind' : True,            # windstress analytical version
            'use_coriolis' : True,        # equatorial beta-plane
            'use_eddy_visc' : True,       # horizonta viscosity
            'use_eddy_diff' : False,      # horizonta viscosity in density
            'use_linfric' : True,         # linear friction
            'use_uniform_mixing' : False, # full version vert. mix.
            'use_non_linear' : False,     # non linear terms
            
        },
        
        #======================================================================
        
        'NlW': {
            # ----------------------manage output------------------------------
            'save_as' : 'NlW',                    # name of output files
            'save_output' : True,                 # saving output (T/F)
            'sample_number' : 100,                # evenly spaced samples
            'av_window' : 58,                     # output average window 
            'zon_sect' : (45, 50, 55),            # loc of zonal sections
            'mer_sect' : (50, 100, 150),          # loc of meridional sect
            'variable_names' : ('u', 'v', 'h', 'w', 'vmix_u', 'vmix_v', 
                                'vmix_h', 'nl_u', 'nl_v', 'nl_h'),         
                                                  # variables to be safed
            # -----------parameters for numerical discretization---------------
            'max_time_step' : 20000,          # number timesteps
            'dt' : 1500,                      # timestep length [s]
            'N_x' : 200,                      # number zonal gridpoints
            'N_y' : 101,                      # number merid. gridpoints
            'N_z' : 900,                      # number vertical gridpoints 
            'L_x' : 5e+6,                     # zonal length of domain [m]
            'L_y' : 2.5e+6,                   # merid. length of domain [m]
            
            # -------------------------run model parallel----------------------
            'domain_split' : (1, 1),             # number of subdomains 
            'nonlinear_split' : 'equation_split',#'equation_split'/'free_split'
            'term_division' : {                  # for free-split only
                'proc1' : ('a1',),
                'proc2' : ('b1',),
                'proc3' : ('c1',),
                'proc4' : ('a2',),
                'proc5' : ('b2',),
                'proc6' : ('c2',),
                'proc7' : ('a3',),
                'proc8' : ('b3',),
                'proc9' : ('c3',),
                'proc10' : ('c4',)
                    }, 
            'swm_split' : 3,                  # number processes SWMint
            'mixing_split' : True,            # strategy vert. mixing
            
            # ------------------boundary conditions----------------------------
            'edge_com' : False,          # communication corner neighbours
            'double_periodic' : False,   # double_periodic boundary condition
            'free_slip' : True,          # free-slip boundary condition
            'halo' : (1, 1),             # number of halo points each side
            
           
            # -------------------physical parameters--------------------------- 
            'mode_list' : np.arange(0, 25, 1),  # list involved modes
            'rho_0' :  1027,                    # reference density [kg/m^3]
            'g' : 10,                           # gravity [m/s^2]
            'phi' : 0,                          # central latitude of domain 
            'omega' : 7.2921150e-5,             # earths rotation [1/s]
            'a' : 6371e+3,                      # radius of earth
            'vis' : 2000,                       # horizontal visc [m^2/s]
            'visd': 0,
            'tau_x_coef' : -0.05/2,             # const. windstress zonal 
            'tau_y_coef' : 0,                   # const. windstress mer.
            'wind_up_t' : 300,
            
            # only used for analytical version of vertical mixing
            'A' : 1.33e-7,                      # const. vert. mix. momentum 
            'B' : 1.33e-7,                      # const. vert. mix density 
            
            # only used for full version of vertical mixing 
            'nu' : 0.001,                       # const. vert. mix. momentum 
            'kap' : 0.0001,                     # const. vert. mix. density
            
            'use_wind' : True,             # windstress analytical version
            'use_coriolis' : True,         # equatorial beta-plane
            'use_eddy_visc' : True,        # horizonta viscosity
            'use_eddy_diff' : False,       # horizonta viscosity in density
            'use_linfric' : True,          # linear friction
            'use_uniform_mixing' : False,  # full version vert. mix.
            'use_non_linear' : True,       # non linear terms
            
        },
        
    #==========================================================================
        
    'NlE': {
            # ----------------------manage output------------------------------
            'save_as' : 'NlE',                   # name of output files
            'save_output' : True,               # saving output (T/F)
            'sample_number' : 100,               # evenly spaced samples
            'av_window' : 58,                    # output average window 
            'zon_sect' : (45, 50, 55),           # loc of zonal sections
            'mer_sect' : (50, 100, 150),         # loc of meridional sect
            'variable_names' : ('u', 'v', 'h', 'w', 'vmix_u', 'vmix_v', 'vmix_h', 
                'nl_u', 'nl_v', 'nl_h'),         # variables to be safed
            # -----------parameters for numerical discretization---------------
            'max_time_step' : 20000,            # number timesteps
            'dt' : 1500,                      # timestep length [s]
            'N_x' : 200,                      # number zonal gridpoints
            'N_y' : 101,                      # number merid. gridpoints
            'N_z' : 900,                      # number vertical gridpoints 
            'L_x' : 5e+6,                     # zonal length of domain [m]
            'L_y' : 2.5e+6,                   # merid. length of domain [m]
            
            # -------------------------run model parallel----------------------
            'domain_split' : (1, 1),          # number of subdomains 
            'nonlinear_split' : 'equation_split',#'equation_split'/'free_split'
            'term_division' : {               # for free-split only
                'proc1' : ('a1',),
                'proc2' : ('b1',),
                'proc3' : ('c1',),
                'proc4' : ('a2',),
                'proc5' : ('b2',),
                'proc6' : ('c2',),
                'proc7' : ('a3',),
                'proc8' : ('b3',),
                'proc9' : ('c3',),
                'proc10' : ('c4',)
                    }, 
            'swm_split' : 3,                  # number processes SWMint
            'mixing_split' : True,            # strategy vert. mixing
            
            # ------------------boundary conditions----------------------------
            'edge_com' : False,          # communication corner neighbours
            'double_periodic' : False,   # double_periodic boundary condition
            'free_slip' : True,          # free-slip boundary condition
            'halo' : (1, 1),             # number of halo points each side
            
           
            # -------------------physical parameters--------------------------- 
            'mode_list' : np.arange(0, 25, 1),  # list involved modes
            'rho_0' :  1027,                    # reference density [kg/m^3]
            'g' : 10,                           # gravity [m/s^2]
            'phi' : 0,                          # central latitude of domain 
            'omega' : 7.2921150e-5,             # earths rotation [1/s]
            'a' : 6371e+3,                      # radius of earth
            'vis' : 3000,                       # horizontal visc [m^2/s]
            'visd': 0,
            'tau_x_coef' : 0.05,                # const. windstress zonal 
            'tau_y_coef' : 0,                   # const. windstress mer.
            'wind_up_t' : 300,
            
            # only used for analytical version of vertical mixing
            'A' : 1.33e-7,                      # const. vert. mix. momentum 
            'B' : 1.33e-7,                      # const. vert. mix density 
            
            # only used for full version of vertical mixing 
            'nu' : 0.001,                       # const. vert. mix. momentum 
            'kap' : 0.0001,                     # const. vert. mix. density
            
            'use_wind' : True,             # windstress analytical version
            'use_coriolis' : True,         # equatorial beta-plane
            'use_eddy_visc' : True,        # horizonta viscosity
            'use_eddy_diff' : False,       # horizonta viscosity in density
            'use_linfric' : True,          # linear friction
            'use_uniform_mixing' : False,  # full version vert. mix.
            'use_non_linear' : True,       # non linear terms
            
        },
    
        
        
    }
    return scenarios


def load( name ):
    scenarios = list_scenarios()
    return scenarios[name]