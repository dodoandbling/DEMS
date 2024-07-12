import numpy as np
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *

from sklearn import metrics
from itertools import combinations
from collections import defaultdict
import copy
import scipy.io as scio

# from pgse_env import pgse_env
# from config.config_mea_idx import define_mea_idx_noise
# from gendata.gen_load import gen_case, gen_load_data

# def get_iqr_data(datas):
#     q1=np.quantile(datas,0.25)
#     q2=np.median(datas)
#     q3=np.quantile(datas,0.75)
#     iqr=q3-q1
#     up=q3+1.5*iqr
#     return up

# def c_bool(c):
#     """
#     The re-estimated injection phase angle c from decomposition：
#     Convert the bus that are attacked to 1, the bus that are not attacked to 0
#     """
#     c_cvt = np.zeros((c.shape[0],c.shape[1]))

#     for i in range(c.shape[0]):
#         # c[i] = [abs(j) for j in c[i]]
#         c_attacked = []
#         c_abs = list(map(abs,c[i]))
#         up = get_iqr_data(c_abs)
#         for j in c_abs:
#             if j > 0.1:
#                 c_attacked.append(1)
#             else:
#                 c_attacked.append(0)
#         c_cvt[i] = c_attacked
#     return c_cvt

# def fpr_tpr(y_true, y_pred):
#     [m,n] = y_true.shape
#     y_true = y_true.reshape(m*n, 1)
#     y_pred = y_pred.reshape(m*n, 1)

#     tn, fp, fn, tp, = metrics.confusion_matrix(y_true, y_pred).ravel()

#     fpr = fp / (fp + tn)
#     tpr = tp / (tp + fn)

#     return fpr,tpr

class mtd():
    def __init__(self, case_env):
        self.no_mea = case_env.no_mea
        # self.ca_sure = np.zeros((times,no_mea))
        self.env = case_env

    def c_est_convert(self,c):
        """
        The re-estimated injection phase angle c from decomposition：
        Convert the bus that are attacked to 1(coordinate is 2, coordinate flag is 3), the bus that are not attacked to 0
        """

        # c = [abs(i) for i in c]
        c_abs = [abs(i) for i in c]
        # c_new.pop(c.index(max(c)))
        # c_new.pop(c.index(max(c)))
        c_attacked = []
        c_dict = defaultdict(list)
        # c_std = np.std(c_new)
        # c_mean = np.mean(c_new)
        # c_abs = list(map(abs,c))
        # up = get_iqr_data(c)
        for i in c_abs:
            if abs(i) >= 0.1:
                c_attacked.append(round(i, 3))
            else:
                c_attacked.append(0)
        for i,j in enumerate(c_attacked):
            c_dict[j].append(i)
        for i in c_dict.keys():                    
            if i!=0:
                if len(c_dict[i])>1:
                    for j in combinations(c_dict[i],2):
                        # diff = abs(c[j[1]]-c[j[0]])
                        if self.env.connected(j[0],j[1]):
                            if self.env.is_leaf(j[0])==0 and c_attacked[j[1]]!=2:
                                c_attacked[j[0]] = 3
                                c_attacked[j[1]] = 2
                            elif self.env.is_leaf(j[1])==0 and c_attacked[j[0]]!=2:
                                c_attacked[j[1]] = 3
                                c_attacked[j[0]] = 2
                        else:
                            if c_attacked[j[0]] != 2 and c_attacked[j[0]] != 0:
                                c_attacked[j[0]] =1
                            if c_attacked[j[1]] != 2 and c_attacked[j[1]] != 0:
                                c_attacked[j[1]] =1
                elif len(c_dict[i])==1:
                    c_attacked[c_dict[i][0]] =1
        return c_attacked
    
    def perturb_strategy(self, c):
        att_bus = []
        for i in range(len(c)):
            if c[i] != 0:
                # att_bus.append(i)
                att_bus.append(self.env.non_ref_index[i])


        brh = [i for i in range(self.env.no_brh)]
        pertub_brh = []
        flag = 0

        for i in range(self.env.no_brh):
            # if self.env.f_bus[i] in att_bus and self.env.t_bus[i] in att_bus:
            #     brh.pop(flag)
            if self.env.f_bus[i] == self.env.ref_index or self.env.t_bus[i] == self.env.ref_index:
                brh.pop(flag)
            else:
                flag = flag + 1
        # print(brh)
        for i in brh:
            if self.env.f_bus[i] in att_bus:
                if c[self.env.non_ref_index.index(self.env.f_bus[i])] !=2:
                    pertub_brh.append(i)
                att_bus.remove(self.env.f_bus[i])
            elif self.env.t_bus[i] in att_bus:
                if c[self.env.non_ref_index.index(self.env.t_bus[i])] !=2:
                    pertub_brh.append(i)
                att_bus.remove(self.env.t_bus[i])
        return pertub_brh
    
    def att_verify(self, c, c_attacked, pertub_brh):
        for i in pertub_brh:
            r, a, a2,se_new,result = self.env.se_mtd(c, i)
            # result = rundcopf(se.case, opt)
            # z, z_noise = self.env.construct_mea(se, result)
            # a = 
            # za2, a2 = gen_fdi(se, z, c)


            if list(a) == list(a2):
            # print(f'branch:{i}')
            # if r < self.env.bdd_threshold:
                if c_attacked[self.env.non_ref_index.index(self.env.f_bus[i])] == 3 or c_attacked[self.env.non_ref_index.index(self.env.t_bus[i])] == 3:
                    for i in c_attacked:
                        if i == 2 or i == 3:
                            i = 0
                elif c_attacked[self.env.non_ref_index.index(self.env.f_bus[i])] == 1:
                    c_attacked[self.env.non_ref_index.index(self.env.f_bus[i])] = 0
                elif c_attacked[self.env.non_ref_index.index(self.env.t_bus[i])] == 1:
                    c_attacked[self.env.non_ref_index.index(self.env.t_bus[i])] = 0
        # return c_attacked, r, env_new
        return c_attacked
    
    
    def att_verify_loop(self, c_true, c_attacked, times):
        c_sure = np.zeros((times,self.env.no_non_ref))
        # r_new_sum = []
        # pbrh_time_sum = []
        for i in range(times):
            # i = 3
            c_attacked_cvt = self.c_est_convert(c_attacked[i])
            pertub_brh = self.perturb_strategy(c_attacked_cvt)
            # pbrh_time_sum.append(len(pertub_brh))
            c_sure_i = self.att_verify(c_true[i], c_attacked_cvt, pertub_brh)
            c_sure[i,:] = c_sure_i
            # r_new_sum.append(r)
        # return c_sure, r_new_sum
        return c_sure
    
    
    def pbtime_loop(self, c_attacked, times):
        pbrh_time_sum = []
        for i in range(times):
            c_attacked_cvt = self.c_est_convert(c_attacked[i])
            pertub_brh = self.perturb_strategy(c_attacked_cvt)
            pbrh_time_sum.append(len(pertub_brh))
        return pbrh_time_sum



# if __name__ == "__main__":
#     # Instance power env
#     case_name = 'case14'
#     case = case14()
#     case = gen_case(case, case_name) 
#     mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'FULL')
#     src_dir = "/Users/dqy/Desktop/code0312/HDS/DCSE/gendata/src"
#     case_env = pgse_env(case = case, case_name = case_name, noise_sigma = noise_sigma, idx = mea_idx, fpr = 0.05)
#     _, _ = gen_load_data(case, case_name, src_dir)

#     noise_flag = 5
#     att_type = 'unmul'
#     att_times = 100
    
#     path = f"/Users/dqy/Desktop/code0312/HDS/data/{case_name}"
#     c_new_sum = scio.loadmat (path+f"/LRMF/c_new/c_new_{att_type}{noise_flag}_sum.mat")['cnew']
#     # original data
#     c_ori_mat = scio.loadmat(path+f"/{att_type}/c_{noise_flag}.mat")['c']
#     c_true_sum = c_ori_mat[:,:,0]

#     c_new_bool = c_bool(c_new_sum)
#     c_true_bool = c_bool(c_true_sum)

#     pb = mtd(case_env=case_env)
#     c_sure_sum = pb.att_verify_loop(c_true_sum,c_new_sum,att_times)
#     # print(c_sure)

#     c_sure_bool = c_bool(c_sure_sum)

#     fp_mtd, tp_mtd = fpr_tpr(c_true_bool, c_sure_bool)
#     fp, tp = fpr_tpr(c_true_bool, c_new_bool)
#     print(f'Case of {att_type} with noise value: {noise_flag}\%')
#     print(f'tp/fp of dual stage scheme:{tp_mtd}/{fp_mtd}')
#     print(f'tp/fp of mf only:{tp}/{fp}')


#     print("hi")


