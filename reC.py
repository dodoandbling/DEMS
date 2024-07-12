import scipy.io as scio
import numpy as np

import matplotlib.pyplot as plt


# Plot options
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['lines.markersize'] = 6
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'normal'
plt.rcParams["axes.labelweight"] = "normal"
plt.rcParams['figure.figsize'] = (5,4)
plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['legend.loc'] = 'lower right'
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['text.usetex'] = True#默认为false，此处设置为TRUE
plt.rcParams['font.family'] = 'Times New Roman'

def absoluteError(a,b):
    # Replace 0 with a minimum value
    # epsilon = 0.01
    aError = a - b
    return aError

path = "/Users/dqy/Desktop/code0312/HDS/data/case14"
H = scio.loadmat(path+"/H.mat")['h']
Wr = scio.loadmat(path+"/W-5.mat")['w']
# Za = sio.loadmat(path+'\Za\za-5%.mat')['za']
Za  = scio.loadmat(path+"/single/za_5.mat")['za'][:,:,0]
A_new = scio.loadmat(path+"/LRMF/a_new/a_new_single5.mat")['A']
Z_new = scio.loadmat(path+"/LRMF/z_new/z_new_single5.mat")['Z']
P_noise = H.T@Wr@H
# P = H.T@H
P_noise_inv = np.linalg.inv(P_noise)
# P_inv = np.linalg.inv(P)

c_new_1 = []
c_new_2 = []

for i in range(100):

    # Weighted Least Squares
    c_est_1 = P_noise_inv@H.T@Wr@A_new[i,:].T
    # c_est_1 = P_inv@H.T@A_new[i,:].T

    # x_ori_est = P_noise_inv@H.T@Wr@Z_ori[i,:].T
    x_new_est = P_noise_inv@H.T@Wr@Z_new[i,:].T
    x_a_est = P_noise_inv@H.T@Wr@Za[i]
    c_est_2 = x_a_est - x_new_est 

    # c_est = P_inv@H.T@A_new[i,:].T

    c_new_1.append(c_est_1)
    c_new_2.append(c_est_2)

# print(c_est)
c_new1 = np.array(c_new_1)
c_new2 = np.array(c_new_2)

# c_new1 = sio.loadmat("E:\MY\paper\FDILocation\code\data\case14\LRMF\c_new1.mat")['c_new1']
# c_new2 = sio.loadmat("E:\MY\paper\FDILocation\code\data\case14\LRMF\c_new2.mat")['c_new2']

ab = absoluteError(c_new1, c_new2)
ab_mean = np.mean(ab,0)

x = range(c_new2.shape[1])
xt = np.arange(2,15)
yt = [0, 1]
plt.ylim(0,0.5)
# plt.plot(xt, ab_mean, color="r", marker='.', linestyle='-')

labelfont = {
    #'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
    'fontsize': 16,
    'fontfamily':'Times New Roman'
}

color_list = ["r","greenyellow","b","orange","c"]
show_list = [7,25,42,74,97]
for i in range(5):
    j= show_list[i]
    y = abs(c_new1[j])
    plt.plot(x, y, color=color_list[i], marker='.', linestyle='-')

plt.grid()
plt.xticks(x)
ax = plt.gca()
ax.ticklabel_format(style='sci', scilimits=(-1,1), axis='y')
ax.set_xticklabels(xt)
# plt.yticks(yt)
plt.xlabel("Bus")  # X轴标签
plt.ylabel("Injected error (in p.u.)")  # Y轴标签
# plt.show()
plt.savefig("/Users/dqy/Desktop/code0312/HDS/pic/c_new2.eps",dpi=600)
# 