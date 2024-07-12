# import numpy as np
import sys, os
# sys.path.insert(0, sys.path[0]+"/../")
sys.path.append("..")

from pypower.api import case118
from pypower.idx_bus import PD, QD
from pypower.idx_brch import RATE_A, BR_X
from config.config_mea_idx import define_mea_idx_noise
from gen_load import gen_case, gen_load_data
from pgse_env import pgse_env
import scipy.io as scio


def dcopf_normal(case_env, times, noise_flag):
    z_sum = []
    # r_sum = []
    print(f'Generating normal data, noise value: {noise_flag}\%')

    for i in range(times):
        result = case_env.run_opf(opf_idx = i)
        # print(f'Is {i}th OPF success: {result["success"]}')
        z, z_noise = case_env.construct_mea(result) # Get the measurement 
        if noise_flag > 0:
            z_mea = z_noise
        else:
            z_mea = z
        z_sum.append(z_mea)
    return z_sum


def dcopf_attack(case_env, times, noise_flag, att_type):
    za_sum = []
    a_sum = []
    c_sum = []
    print(f'Generating attack data, attack type: {att_type}, noise value: {noise_flag}\%')

    for i in range(times):
        result = case_env.run_opf(opf_idx = i+5)
        # print(f'Is OPF success: {result["success"]}')
        z, z_noise = case_env.construct_mea(result) # Get the measurement
        if noise_flag > 0:
            z_mea = z_noise
        else:
            z_mea = z
        
        if att_type == 'single':#single-bus
            za_fdi,a,c = case_env.gen_sin_fdi(z_mea)
        elif att_type == 'unmul': #uncoordiante multiple-bus
            za_fdi,a,c = case_env.gen_mul_fdi(z_mea)
        elif att_type == 'comul': #coordiante multiple-bus
            za_fdi,a,c = case_env.gen_co_fdi(z_mea)

        za_sum.append(za_fdi)
        a_sum.append(a)
        c_sum.append(c)
    return za_sum, a_sum, c_sum



if __name__ == "__main__":
    # Instance power env
    case_name = 'case118'
    case = case118()
    case = gen_case(case, 'case118')  # Modify the case
    
    # Define measurement index
    mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'FULL')
    
    # Generate load if it does not exist
    src_dir = "src"
    _, _ = gen_load_data(case, 'case118', src_dir)
    
    # Instance the class
    case_env = pgse_env(case = case, case_name = case_name, noise_sigma = noise_sigma, idx = mea_idx, fpr = 0.05)
    
    # data_config
    att_flag = 1
    noise_flag = 5
    att_type = 'unmul'
    att_times = 100
    
    # saving_path
    z_dir = f"../../data/{case_name}/z_{noise_flag}.mat"
    za_dir = f"../../data/{case_name}/{att_type}/za_{noise_flag}.mat"
    a_dir = f"../../data/{case_name}/{att_type}/a_{noise_flag}.mat"
    c_dir = f"../../data/{case_name}/{att_type}/c_{noise_flag}.mat"

    if att_flag:
        za_sum, a_sum, c_sum = dcopf_attack(case_env, att_times, noise_flag, att_type)
        scio.savemat(za_dir, {'za': za_sum})
        scio.savemat(a_dir, {'a': a_sum})
        scio.savemat(c_dir, {'c': c_sum})
    else:
        z_sum = dcopf_normal(case_env, att_times, noise_flag)
        scio.savemat(z_dir, {'z': z_sum})
    
    

    # scio.savemat(f"../../data/{case_name}/H.mat", {'h': case_env.H})
    # scio.savemat(f"../../data/{case_name}/W-{noise_flag}.mat", {'w': case_env.Wr})
    # # 
    




