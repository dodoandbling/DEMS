"""
A simple realization of MATPOWER DC state estimation and false data injection using PyPower

Author: QINGYUN DU
"""

from turtle import shape
import numpy as np
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *
# from config.config_mea_idx import define_mea_idx_noise
# from config.config_se import se_config, opt
import copy
from scipy.stats.distributions import chi2
# from gendata.gen_load import gen_case, gen_load_data
# import scipy.io as scio
import random

class pgse_env:
    def __init__(self, case, case_name, noise_sigma, idx, fpr):
        
        """
        case: the instances case by calling from pypower api, e.g. case = case14()
        noise_sigma = A 1D array contains the noise std of the measurement, please refer to the format in mea_idx 
        idx: the measurement index given by each measurement type, please refer to the format in mea_idx
        
        measurement type (the order matters)
        z = [pi, pf, -pf]  (in our current settings)
        """

        """
        Define the grid parameter
        """

        # Case
        self.case = case
        case_int = ext2int(case)                                                         # Covert the start-1 to start-0 in python
        self.case_int = case_int

        # Determine the bus type
        self.ref_index, pv_index, pq_index = bustypes(case_int['bus'], case_int['gen'])  # reference bus (slack bus), pv bus, and pq (load bus)
        self.non_ref_index = list(pq_index) + list(pv_index)                             # non reference bus
        self.non_ref_index.sort()

        # Numbers
        self.no_bus = len(case['bus'])
        self.no_brh = len(case['branch'])
        self.no_gen = len(case['gen'])
        self.no_non_ref = len(self.non_ref_index)
        self.no_mea = self.no_non_ref + 2*self.no_brh

        self.f_bus = self.case_int['branch'][:, 0].astype('int')        # list of "from" buses
        self.t_bus = self.case_int['branch'][:, 1].astype('int')        # list of "to" buses
        
        """
        Define matrices related to measurement noise
        """
        self.noise_sigma = noise_sigma                     # std
        self.R = np.diag(noise_sigma**2)                   # R
        R = np.delete(self.R, self.ref_index, axis = 1) 
        self.Rr = np.delete(R, self.ref_index, axis = 0)   # the reference bus reduced R

        self.W = np.diag(1/self.noise_sigma**2)        # R^-1->W
        W = np.delete(self.W, self.ref_index, axis = 1) 
        self.Wr = np.delete(W, self.ref_index, axis = 0)   # the reference bus reduced W
        
        DoF = self.no_mea - (self.no_bus - 1)            # Degree of Freedom
        self.bdd_threshold = chi2.ppf(1-fpr, df = DoF)    # BDD detection threshold
        # self.bdd_threshold = chi2.ppf(1-fpr, df = DoF)  
          
        """
        Incidence Matrix
        """
        # Generator Incidence Matrix
        self.Cg = np.zeros((self.no_bus,self.no_gen))
        for i in range(self.no_gen):
            self.Cg[int(case_int['gen'][i,0]),i] = 1
        
        # Branch Incidence Matrix
        self.f_bus = case_int['branch'][:, 0].astype('int')        # list of "from" buses
        self.t_bus = case_int['branch'][:, 1].astype('int')        # list of "to" buses
        # print(self.f_bus)
        self.Cf = np.zeros((self.no_brh,self.no_bus))         # "from" bus incidence matrix
        self.Ct = np.zeros((self.no_brh,self.no_bus))         # "to" bus incidence matrix
        for i in range(self.no_brh):
            self.Cf[i,self.f_bus[i]] = 1
            self.Ct[i,self.t_bus[i]] = 1

        self.A = self.Cf - self.Ct                                # the full incidence matrix
        self.Ar = np.delete(self.A, self.ref_index, axis = 1)     # the reduced incidence matrix     
        
        self.x = case_int['branch'][:,3]                          # reactance
        self.neg_b = -1/self.x

        # susceptance matrix
        self.B = self.Ar.T@np.diag(self.neg_b)@self.Ar
        self.S = np.diag(self.neg_b)@self.Ar

        # measurement matrix
        self.H = np.vstack((self.B,np.vstack((self.S,-self.S))))

        """
        MTD settings
        """
        self.max_reac_ratio = 0.5
        self.min_reac_ratio = 0.2

        # Test the case
        result = self.run_opf()
        if result['success']:
            print('Initial OPF tests ok.')
        else:
            print('The OPF under this load condition is not converged! Decrease the load.')

        print('*'*60)

        # Load
        load_active_dir = f"/Users/dqy/Desktop/code0312/HDS/DCSE/gendata/src/load/load_active_{case_name}.npy"
        load_reactive_dir = f"/Users/dqy/Desktop/code0312/HDS/DCSE/gendata/src/load/load_reactive_{case_name}.npy"

        self.load_active = np.load(load_active_dir)
        self.load_reactive = np.load(load_reactive_dir)        

        # The default reactance in case file
        self.reactance_ori = copy.deepcopy(self.case['branch'][:,BR_X])   


    def update_H(self, brh):
        """
        Update H of self
        """
        increase_decrease = np.random.randint(0, 2)*2-1   # -1 or 1
        # ratio = np.ones(self.no_brh,)
        ratio = (self.min_reac_ratio + (self.max_reac_ratio - self.min_reac_ratio) * np.random.rand())*increase_decrease
        # ratio = 8
        self.x[brh] = self.x[brh] * (1+ratio)

        self.case['branch'][:,BR_X] = self.x
        self.case_int = ext2int(self.case)
        
        # reactance
        self.neg_b = -1/self.x

        # susceptance matrix
        self.B = self.Ar.T@np.diag(self.neg_b)@self.Ar
        self.S = np.diag(self.neg_b)@self.Ar

        # measurement matrix
        self.H = np.vstack((self.B,np.vstack((self.S,-self.S))))
    
    def run_opf(self, **kwargs):
        """
        The run_opf function to directly read OPF index
        """
        
        case_opf = copy.deepcopy(self.case)
        if 'opf_idx' in kwargs.keys():
            # print(f'Load index: {10*kwargs["opf_idx"]}.')
            # Specify the index
            load_active_ = self.load_active[int(10*kwargs['opf_idx'])]
            load_reactive_ = self.load_reactive[int(10*kwargs['opf_idx'])]
        
            case_opf['bus'][:,PD] = load_active_
            case_opf['bus'][:,QD] = load_reactive_
        else:
            # run the default
            print('Run on the default load condition.')
            pass
        
        opt = ppoption()              # OPF options
        opt['VERBOSE'] = 0
        opt['OUT_ALL'] = 0
        opt['OPF_FLOW_LIM'] = 1
        result = rundcopf(case_opf, opt)
        
        return result

    def construct_mea(self, result):
        """
        Given the OPF result, construct the measurement vector
        z = [pi, pf, -pf] in the current setting
        """
        pf = result['branch'][:,PF]/self.case['baseMVA']
        pi = (self.Cg@result['gen'][:,PG] - result['bus'][:,PD])/self.case['baseMVA']
        pir = np.delete(pi, self.ref_index, axis = 0)     # the reference bus reduced measurement

        z = np.concatenate([pir, pf, -pf], axis = 0)

        z_noise = z + np.random.multivariate_normal(mean = np.zeros((self.no_mea,)), cov = self.Rr)
        z = np.expand_dims(z, axis = 1)
        z_noise = np.expand_dims(z_noise, axis = 1)
        
        return z, z_noise

    def dc_se(self, z_noise):
        """
        Solve for state variables x_est using weighted least squares
        Estimate the measurement from the state: z_est = H·x_est
        BDD: Find the residual of chi^2 detector given the estimated state
        """

        P_noise = self.H.T@self.Wr@self.H
        # P = self.H.T@self.H
        P_noise_inv = np.linalg.inv(P_noise)
        # P_inv = np.linalg.inv(P)
        
        # Weighted Least Squares
        x_est = P_noise_inv@self.H.T@self.Wr@z_noise
        # x_est = P_inv@self.H.T@z_noise

        # Find z_est
        z_est = self.H@x_est

        # Find the residual of chi^2
        # r_chi = np.linalg.norm(r, ord = 2)
        r_chi = ((z_noise-z_est).T@self.Wr@(z_noise-z_est))[0,0]

        return x_est, z_est, r_chi

    
    def gen_ran_att(self, z_noise):
        """
        Generate a random attack without using the knowledge of model
        att_ratio_max: the maximum change ratio of each measurement
        """
        att_ratio_max = 0.5
        att_ratio = -att_ratio_max + att_ratio_max*2*np.random.rand(z_noise.shape[0])
        att_ratio = np.expand_dims(att_ratio, axis = 1)
        z_att_noise = z_noise * (1+att_ratio)
        # a = np.zeros((self.no_mea, 1))
        # for i in range(4):
        #     a[i+8] = 6
        # z_att_noise = z_noise + a
        return z_att_noise    
    
    def gen_sin_fdi(self, z_noise):
        """
        Single bus / random value / FDI attack
        """

        # The injected offset on the system states
        c = np.zeros((self.no_non_ref, 1))
        # r_p = random.randint(0, self.no_non_ref-1)
        # ang_posi = self.non_ref_index[r_p]
        ang_posi = random.randint(0, self.no_non_ref-1)

        # ang_posi = 2
        while abs(c[ang_posi]) < 0.1:
            c[ang_posi]= (0.5-(-0.5)) * np.random.random() + (-0.5) 
        # Attack vector
        a = self.H@c
        z_a = z_noise + a

        return z_a, a, c

    def gen_mul_fdi(self, z_noise):
        """
        Multi bus / random value / FDI attack
        """

        # The injected offset on the system states
        # The injected offset on the system states
        c = np.zeros((self.no_non_ref, 1))
        attack_num =  random.randint(2, 6)
        # att_bus  = [2,4]
        for i in range(attack_num):
            j = random.randint(0, self.no_non_ref-1)
            # j = att_bus[i]
            while abs(c[j]) < 0.1:
                c[j]= (0.5-(-0.5)) * np.random.random() + (-0.5)

        # Attack vector
        a = self.H@c
        z_a = z_noise +a

        return z_a, a, c

    def gen_co_fdi(self, z_noise):
        """
        Meter Targetted / random value / FDI attack
        """
        att_bus = []
        attack_num =  random.randint(2, 6)
        att_bus.append(self.non_ref_index[random.randint(0, self.no_non_ref-1)])
        for i in range(attack_num-1):
            for j in att_bus:
                if j in list(self.f_bus):
                    ind = list(self.f_bus).index(j)
                    b = self.t_bus[ind]
                    if b in att_bus:
                        pass
                    else:
                        att_bus.append(b)
                        break
                elif j in list(self.t_bus):
                    b = self.f_bus[list(self.t_bus).index(j)]
                    if b in att_bus:
                        pass
                    else:
                        att_bus.append(b)
                        break
        # The injected offset on the system states
        c = np.zeros((self.no_non_ref, 1))
        temp = 0
        while abs(temp) < 0.1:
            temp = (0.5-(-0.5)) * np.random.random() + (-0.5)
        for i in att_bus:
            # print(i)
            j = int(i)
            if j >self.ref_index :
                j=j-1
            c[j]=  temp

        # Attack vector
        a = self.H@c
        z_a = z_noise +a

        return z_a, a, c
    
    def connected(self,a,b):
        f = np.where(self.f_bus == a)[0]
        t = np.where(self.t_bus == a)[0]

        if len(f) != 0:
            for i in f:
                if b == self.t_bus[i]:
                    return 1
        if len(t) != 0:
            for i in t:
                if b == self.f_bus[i]:
                  return 1
        return 0

    def is_leaf(self,node):
        node_list = list(self.f_bus) + list(self.t_bus)
        flag = node_list.count(node)
        if flag == 1:
            return 1
        else:
            return 0
        
    def se_mtd(self, c, pertub_brh):
        se = copy.deepcopy(self)
        se.update_H(pertub_brh)
        # opt = ppoption()              # OPF options
        # opt['VERBOSE'] = 0
        # opt['OUT_ALL'] = 0 
        # opt['OPF_FLOW_LIM'] = 1

        result = se.run_opf(opf_idx=2)
        z, z_noise = se.construct_mea(result)
        a = se.H@c
        a2 = self.H@c
        z_a = np.add(z_noise,a2.reshape(self.no_mea,1))
        # print(z)

        _, _, r = se.dc_se(z_a)
        # print(z_a)


        return  r, a, a2, se, result
    
# """
# Test
# """
# if __name__ == "__main__":
#     # Instance power env
#     case_name = 'case39'
#     case = case39()
#     case = gen_case(case, 'case39')  # Modify the case

#     # Generate load if it does not exist
#     _, _ = gen_load_data(case, 'case39')
    
#     # Define measurement index
#     mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'FULL')
    
#     # Instance the class
#     case_env = pgse_env(case = case, case_name = case_name, noise_sigma = noise_sigma, idx = mea_idx, fpr = 0.05)
    

    
#     # Run opf 
#     result = case_env.run_opf()
    
#     # Construct the measurement
#     z, z_noise = case_env.construct_mea(result) # Get the measurement
    
#     # Run DC-SE     
#     x_est, z_est, r = case_env.dc_se(z_noise)

#     # BDD   
#     print(f'BDD threshold: {case_env.bdd_threshold}')
#     print(f'residual: {r}')

#     print('*'*60)
#     print('Run OPF on a given load')
#     # Run OPF on a given load
#     result = case_env.run_opf(opf_idx = 1)
#     gc = result['f']
#     # Construct the measurement
#     z, z_noise = case_env.construct_mea(result) # Get the measurement   
#     x_est, z_est, r = case_env.dc_se(z_noise)
#     print(f'BDD threshold: {case_env.bdd_threshold}')
#     print(f'residual without attack: {r}')
#     # print(f'generator cost without MTD: {gc}')

#     # Run single-bus FDI attack
#     print('*'*60)
#     print('Run single-bus FDI attack')
#     za_fdi,a,c = case_env.gen_sin_fdi(z_noise)

#     # path = "E:\\MY\\paper\\FDILocation\\code\\data\\case14\\single"
#     # c_ori_mat = scio.loadmat(path+"\c_0.mat")['c']
#     # c_ori = c_ori_mat[:,:,0]
#     # c = c_ori[0]
#     # a = case_env.H@c
#     # za_fdi = np.add(z_noise,a.reshape(53,1))
#     x_fdi, z_fdi, r_fdi = case_env.dc_se(za_fdi)

#     print(f'residual with FDI attack: {r_fdi}')

#     print('*'*60)
#     print('Run MTD')
#     # Run MTD
#     pos = np.argmax(abs(c))
#     attack_bus = case_env.non_ref_index[pos]
#     brh = 0
#     for i in range(case_env.no_brh): # Find the branch connected to the attacked bus
#         f = case_env.f_bus[i]
#         t = case_env.t_bus[i]
#         if f==attack_bus or t==attack_bus:
#             brh = i
#             break
#     se, r_mtd, result_mtd = case_env.se_mtd(c,brh)
#     gc_mtd = result_mtd['f']
#     # se, r_mtd = case_env.se_mtd(c,4)
#     print(f'residual with FDI attack and MTD: {r_mtd}')
    # print(f'generator cost after MTD: {gc_mtd}')
    # path = "E:\MY\paper\FDILocation\code\data\case14"
    # z_sum5 = scio.loadmat(path+"\\z_0.mat")['z'][:,:,0]
    # z =np.expand_dims(z_sum5[1], axis = 1)
    # x_est, z_est, r = case_env.dc_se(z)
    # print(r)
