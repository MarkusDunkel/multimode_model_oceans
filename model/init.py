#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 15:12:08 2019

@author: Markus Dunkel
"""
import numpy as np
import vertical_structure_functions.dynmodes as dyn 
from scipy import interpolate
import os.path
import compute_triple_products_of_modes as tr
import variables as var
import scenarios


def interpolate_to_model_levels( wmodes, pmodes, c, dz, depth, Nsq, level_num):
    Nsq_inter = interpolate.interp1d( depth, Nsq, kind='cubic')
                   
    wmodes_inter = [0] * len(wmodes)               
    for i in range(len(wmodes)):
        wmodes_inter[i] = interpolate.interp1d( 
            depth, wmodes[i,:], kind ='cubic')
     
    depthp = np.linspace(depth[0]-dz[0][0], depth[-1]+dz[0][0], len(depth)-1)
    pmodes_inter = [0] * len(pmodes)               
    for i in range(len(pmodes)):
        pmodes_inter[i] = interpolate.interp1d( depthp, pmodes[i,:] )
        
    depthw_new = np.linspace(depth[0], depth[-1], level_num)
    dz_new = depthw_new[0] + depthw_new[1]
    depthp_new =np.linspace(depth[0]+dz_new/2, depth[-1]-dz_new/2, level_num-1)
    Nsq_new = Nsq_inter(depthw_new)
    
    wmodes_new = np.zeros((len(wmodes), len(depthw_new)))
    for i in range(len(wmodes)):
        wmodes_new[i, :] = wmodes_inter[i](depthw_new)
    
    pmodes_new = np.zeros((len(pmodes), len(depthp_new)))
    for i in range(len(pmodes)):
        pmodes_new[i, :] = pmodes_inter[i](depthp_new)
    
    wdz_new = np.full((1, level_num), dz_new)
    wdz_new[0, -1] = dz_new/2
    wdz_new[0, 0] = dz_new/2
    pdz_new = np.full((1, level_num-1), dz_new)
      
    return Nsq_new, wmodes_new, pmodes_new, depthw_new, depthp_new, wdz_new, \
        pdz_new
    

def interpolate_Nsq( Nsq, depth ):
    
    Nsq_f = np.flip(Nsq)
    depth_f = np.flip(depth)
    arr_l = len(Nsq)
    
    weight = np.linspace(1e6, 1, arr_l)
    weight[0] = 1e7
    
    tck =interpolate.splrep(depth_f, Nsq_f, weight, xb=None, xe=None, k=5, s=3)
    
    depth_fe=np.linspace(depth_f[0], depth_f[-1], (depth_f[-1] - depth_f[0])/5)
    
    Nsq_fe = interpolate.splev(depth_fe, tck, der=0)
    Nsq_e = np.flip(Nsq_fe)
    depth_e = np.flip(depth_fe)
    
    return Nsq_e, depth_e


def compute_vertical_structure_functions( number_of_modes ):
    
    Nsq, depth = np.load( "Nsq_dataset/Nsq.npy")
    
    # use constant stratification
    use_const_strat = False
    if use_const_strat:
        Nsq *= 0
        Nsq += 3.497e-6
    
    # profile is non regular, so it is interpolated first
    Nsq_regular, depth_regular = interpolate_Nsq( Nsq, depth )
    
    # computation of vertical structure functions
    modes, c, dz  = dyn.dynmodes(
        Nsq_regular, depth_regular, number_of_modes, normalize = True)
    
    # normalize w-modes 
    # remove normalization mistake for the w functions
    norm_w = False
    if norm_w:
        shape = np.shape(modes[0])
        for i in range(shape[0]):
            for j in range(shape[1]):
                modes[0][i, j] *= 10 / c[0][i]
    
    p = 'vertical_structure_functions/vertical_structure_functions_dataset.npy'
    np.save( p, (modes, c, dz, depth_regular, Nsq_regular) )
    
    
class MetaData:
    
    def __init__(self):
        
#==============================================================================
#------------------------ Start Initialization by User ------------------------
#============================================================================== 
        '''
        Each model run saves the compiled metadata in a result folder on the 
        same instance as the folder with the model code. If metadata should be 
        saved as a scenario it needs to be moved inside of the model folder 
        and stored in the scenario folder. 
        
        Then, either a scenario can be loaded or initialization can be done 
        with a new model set up. If a scenario is loaded, modifications can be 
        done in the -changes to loaded scenario- section. 
        
        '''
        load_scenario = True
        scenario_name = 'LW'
        if load_scenario: 
            self.__dict__ = scenarios.load(scenario_name)
              
        else: 
            print('Initialization with new model set up.')
            
            # ----------------------manage output------------------------------
            self.save_as = 'some_name'                # name of output files
            self.save_output = False                  # saving output (T/F)
            self.sample_number = 150                  # evenly spaced samples
            self.av_window = 1                        # output average window 
            self.zon_sect = (45, 50, 55)              # loc of zonal sections
            self.mer_sect = (50, 100, 150)            # loc of meridional sect
            names = ('u', 'v', 'h', 'w', 'vmix_u', 'vmix_v', 'vmix_h', 'nl_u', 
                     'nl_v', 'nl_h') 
            self.variable_names = names               # fields saved in output 
                                                 
            # -----------parameters for numerical discretization---------------
            self.max_time_step = 1000           # number timesteps
            self.dt = 1500                      # timestep length [s]
            self.N_x = 200                      # number zonal gridpoints
            self.N_y = 101                      # number merid. gridpoints
            self.N_z = 500                      # number vertical gridpoints 
            self.L_x = 5e+6                     # zonal length of domain [m]
            self.L_y = 2.5e+6                   # merid. length of domain [m]
            
            # -------------------------run model parallel----------------------
            self.domain_split = (1, 1)                # number of subdomains 
            nonlinear_split = ('equation_split', 'free_split')
            self.nonlinear_split = nonlinear_split[0] # strategy nonlinear term
            if self.nonlinear_split == 'free_split':
                self.term_division = {
                    'proc1' : ('a1', 'a2'),
                    'proc2' : ('b1', 'b2'),
                    'proc3' : ('c1', 'c2'),
                    'proc4' : ('a3', 'b3'),
                    'proc5' : ('c3', 'c4'),
                    }  
            self.swm_split = 3                        # number processes SWMint
            self.mixing_split = True                  # strategy vert. mixing
            
            # ------------------boundary conditions----------------------------
            self.edge_com = False          # communication corner neighbours
            self.double_periodic = False   # double_periodic boundary condition
            self.free_slip = True          # free-slip boundary condition
            self.halo = (1, 1)             # number of halo points each side
            
           
            # -------------------physical parameters--------------------------- 
            self.mode_list = np.arange(0, 25, 1)   # list involved modes
            self.rho_0 =  1027                    # reference density [kg/m^3]
            self.g = 10                           # gravity [m/s^2]
            self.phi = 0                          # central latitude of domain 
            self.omega = 7.2921150e-5             # earths rotation [1/s]
            self.a = 6371e+3                      # radius of earth
            self.vis = 2000                       # horizontal visc [m^2/s]
            self.visd = 2000                      # hor vis in density  [m^2/s]
            self.tau_x_coef = -0.05               # const. windstress zonal 
            self.tau_y_coef = 0                   # const. windstress mer.
            self.wind_up_t = 300
            
            # only used for analytical version of vertical mixing
            self.A = 1.33e-7                      # const. vert. mix. momentum 
            self.B = 1.33e-7                      # const. vert. mix density 
            
            # only used for full version of vertical mixing 
            self.nu = 0.001                       # const. vert. mix. momentum 
            self.kap = 0.0001                     # const. vert. mix. density
            
            self.use_wind = True            # windstress analytical version
            self.use_coriolis = True        # equatorial beta-plane
            self.use_eddy_visc = True       # horizonta viscosity
            self.use_linfric = True         # linear friction
            self.use_uniform_mixing = False # full version vert. mix.
            self.use_non_linear = True       # non linear terms
        
#==============================================================================
#------------------------ End Initialization by User --------------------------
#============================================================================== 
        
        if self.free_slip == True and self.edge_com == True:
            print('Slip boundary condition and communication with' +
                  'corner neighbours is not implemented.')
            
        if self.free_slip == True and self.double_periodic == True:
            print('Special treatment of boundaries is not needed since' +
                  'boundaries are disabled. >>no_slip<< is set True.')
            self.no_slip == True
            
#---------------------------- automatic generated parameters ------------------
        self.dx = self.L_x/(self.N_x -0.5)                       # grid spacing
        self.dy = self.L_y/(self.N_y -0.5)                       # grid spacing
        self.x = np.linspace(0, self.L_x, self.N_x)              # arr x-axis
        self.y = np.linspace(self.L_y/2, -self.L_y/2, self.N_y)  # arr y-axis
        self.mode_num = len(self.mode_list)
        
        if (self.save_output is True):  
            self.av_steps = np.arange(
                self.av_window-1, self.max_time_step, self.av_window)
            self.step_len = int(len(self.av_steps)/self.sample_number)
            self.save_time_steps =self.av_steps[self.step_len-1::self.step_len]
        
        self.tau_x = np.ones( (self.N_y, self.N_x) ) * self.tau_x_coef   
        self.tau_y = np.ones( (self.N_y, self.N_x) ) * self.tau_y_coef
        
        self.f_on_u = 0
        self.f_on_v = 0
        if (self.use_coriolis is True):
            self.get_f_values_on_grid_points()
       
#-------------------------- load or compute modes------------------------------
        # smoothing of the stratification dataset
        wmodes, pmodes, self.c, self.dz_w, self.dz_p, self.depth_w, \
            self.depth_p, Nsq = self.load_modes()
            
        self.pmodes = var.Profile(self.mode_num, self.N_z, pmodes, 'p', True)
        self.wmodes = var.Profile(self.mode_num, self.N_z, wmodes, 'w', True)
        self.Nsq = var.Profile(self.mode_num, self.N_z, Nsq, 'w', False)
         
        self.dz_int = self.dz_p[0, 0]
        
#------------------------compute mode dependent paramters----------------------
        #compute mode dependent paramters such as equivalent depth
        self.H = np.zeros(( self.mode_num ))
        self.pmode_null = np.zeros(( self.mode_num ))
        for i in range( self.mode_num ):
            self.H[i] = self.c[i]**2 / self.g 
            self.pmode_null[i] = self.pmodes._on_p[i][0]
        
#---------------------- load or compute tri-products of modes------------------
        if self.use_non_linear:
            self.load_triple_products_of_modes()
            
#-schedule for organizing SWMs into bunches for computing serially in workers--
        self.swm_keys = [0] * self.swm_split
        for i in range(self.swm_split):
            self.swm_keys[i] = np.arange(i, self.mode_num, self.swm_split)
        
    def load_modes(self):
        """Vertical structure functions are loaded. If they are not stored, 
        they are computed from Nsq_dataset first. 
        """
        
        adress = os.path.isfile( 
             'vertical_structure_functions/' + \
             'vertical_structure_functions_dataset.npy' )
         
        if not adress:
            n = 50 # compute first n modes 
            compute_vertical_structure_functions( n )
            
        modes, c, dz, depth, Nsq = np.load( 
            'vertical_structure_functions/' +  \
            'vertical_structure_functions_dataset.npy', 
            allow_pickle=True)
        
        Nsq_ml, wmodes_ml, pmodes_ml, depthw_ml, depthp_ml, wdz_ml, pdz_ml = \
            interpolate_to_model_levels( 
                modes[0], modes[1], c, dz, depth, Nsq, self.N_z )
        
        if self.N_z < 200:
            print('Analysis has shown that important features of the strati'+\
                  'fication might be unresolved with a vertical grid number '+\
                  'smaller than 200.')
                
        pdz_ml *= -1
        wdz_ml *= -1
        
        return wmodes_ml[self.mode_list, :], pmodes_ml[self.mode_list, :], \
            c[0][self.mode_list], wdz_ml, pdz_ml, depthw_ml, depthp_ml, Nsq_ml

            
    def load_triple_products_of_modes(self):
        """Triple products of modes are loaded. If they are not stored, they 
        are computed first. 
        """
        #adress = os.path.isfile('triple_mode_tensors/PPP.npy' )
         
        #if not adress:
        mode_tensors = tr.compute_triple_mode_tensors(
        self.pmodes, self.wmodes, self.mode_num, self.dz_int, self.N_z, 
        self.Nsq, self.rho_0, self.g )
            
        self.PPP, self.PPWz, self.WPWdz, self.WWWzdz, self.WWWdzz =mode_tensors
            
            
    def get_f_values_on_grid_points(self):
        self.beta= 2 * self.omega / self.a * np.cos(self.phi / 360 * 2 * np.pi)        
        self.f_0 = 2 * self.omega * np.sin(self.phi / 360 * 2 * np.pi)
        
        if self.N_y % 2 == 0:
            self.double_y_t = np.linspace(
                self.L_y/2 + self.dy/4,  -self.L_y/2 + self.dy/4,  self.N_y*2)
        else:
            self.double_y_t = np.linspace(
                self.L_y/2 - self.dy/4,  -self.L_y/2 - self.dy/4,  self.N_y*2)
            
        f_vec_t = self.f_0 + self.double_y_t * self.beta
        f_mat_t = np.repeat(f_vec_t[:, np.newaxis], self.N_x * 2, 1)
        
        self.f_on_u = f_mat_t[1::2, 0::2]
        self.f_on_v = f_mat_t[0::2, 1::2]
