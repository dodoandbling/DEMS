import time

from godec import godec
import scipy.io as sio
from utils import play_2d_results, play_2d_video

from numpy import prod, zeros, sqrt
from numpy.random import randn
from scipy.linalg import qr
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import numpy as np



def godec(X):
    # t = time.time()
    rank=5
    card=None
    iterated_power=20
    max_iter=1000
    tol=0.0000001

    iter = 1
    RMSE = []
    card = prod(X.shape) if card is None else card

    X = X.T if(X.shape[0] < X.shape[1]) else X
    m, n = X.shape

    L = X
    S = zeros(X.shape)
    LS = zeros(X.shape)
    start = time.time()

    while True:
        # Update of L
        Y2 = randn(n, rank)
        for i in range(iterated_power):
            Y1 = L.dot(Y2)
            Y2 = L.T.dot(Y1)
        Q, R = qr(Y2, mode='economic')
        L_new = (L.dot(Q)).dot(Q.T)

        # Update of S
        T = L - L_new + S
        L = L_new
        T_vec = T.reshape(-1)
        S_vec = S.reshape(-1)
        idx = abs(T_vec).argsort()[::-1]
        S_vec[idx[:card]] = T_vec[idx[:card]]
        S = S_vec.reshape(S.shape)

        # Reconstruction
        LS = L + S

        # Stopping criteria
        error = sqrt(mean_squared_error(X, LS))
        RMSE.append(error)

        # print("iter: ", iter, "error: ", error)
        if (error <= tol) or (iter >= max_iter):
            break
        else:
            iter = iter + 1
    t = time.time() - start
    return L,S,t

if __name__ == "__main__":

    casename = 'case39'
    noise_flag = 5
    att_type = 'unmul'

    # Load data
    path = f"XXX\data\\{casename}"
    mat_z = sio.loadmat(path+f"\\z_{noise_flag}.mat")
    mat_za = sio.loadmat(path+f"\\{att_type}\za_{noise_flag}.mat")
    z = mat_z['z'][:,:,0]
    za = mat_za['za'][:,:,0]

    [att_times, no_mea] = za.shape

    A = zeros((att_times, no_mea))
    Z = zeros((att_times, no_mea))
    T = []
    e = np.vstack((z,z))
    window = 10
    t = e[1:window-1,:]
    z_w = z[0:-1,:]
    for i in range(att_times):
        N = np.vstack((z_w,za[i,:]))
        M = N.T

        [m,n] = M.shape
        L,S,t = godec(M)

        Z[i,:] = S.T[-1,:]
        A[i,:] = L.T[-1,:]
        T.append(t)
        # plt.imshow(L)
        # plt.imshow(S)
        # sio.savemat(path+'\A_new\godec\S_new-10-cs.mat', {'S': L})
    sio.savemat(path+f'\\GoDEC\\a_new\\a_new_{att_type}{noise_flag}.mat', {'A': A})
    sio.savemat(path+f'\\GoDEC\\z_new\\z_new_{att_type}{noise_flag}.mat', {'Z': Z})
    # meantime = np.mean(T)
    # timestr = f'{casename}:number of measurement:{no_mea}/{att_type}{noise_flag}/mean cpu time:{meantime}'
    # with open(path+f'\\GoDEC\\time.txt','a+') as f:    #设置文件对象
    #     f.write(timestr)                 #将字符串写入文件中

    # sio.savemat(path+'\A_new\godec\A_new-10-cs.mat', {'A': A})
    # sio.savemat(path+'\Z_new\godec\Z_new-10-cs.mat', {'Z': Z})


