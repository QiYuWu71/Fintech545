import pandas as pd
import numpy as np
from scipy.linalg import fractional_matrix_power

#Cholskey PSD
def chol_psd(sigma):
    n = sigma.shape[0]
    root = np.zeros((n,n))
    for j in range(n):
        s = root[j,:j]
        diff = sigma[j,j] - np.sum(s*s)
        if diff <=0 and diff>= -1e-8:
            diff = 0
        root[j,j] = np.sqrt(diff)
        if root[j,j] == 0:
            root[j,(j+1):n] = 0
        else:
            ir = 1/root[j,j]
            for i in range(j+1,n):
                if j ==0:
                    s = 0
                else:
                    s = np.sum(root[i,:j]*root[j,:j])
                root[i,j] = (sigma[i,j]-s)*ir
    return root

# Near_PSD
def near_psd(sigma,eps):

    # calculate the correlation matrix
    n = sigma.shape[0]
    inv_var = np.zeros((n,n))
    for i in range(n):
        inv_var[i,i] = 1/sigma[i,i]
    corr = inv_var@sigma@inv_var

    # SVD, update the eigen value and scale
    vals,vecs = np.linalg.eig(corr)
    vals[vals<eps] = eps
    T = 1/ (vecs*vecs@vals)
    T = np.diag(np.sqrt(T).tolist())
    l = np.diag(np.sqrt(vals).tolist())
    B = T@vecs@l
    out = B@B.T

    # back to the variance matrix
    var_mat = np.zeros((n,n))
    for i in range(n):
        var_mat[i,i] = sigma[i,i]
    cov = var_mat@out@var_mat

    return cov

#  HigHam
def eigen_con(X):
    n = X.shape[0]
    W = np.identity(n)
    A = fractional_matrix_power(W,0.5)@X@fractional_matrix_power(W,0.5)
    vals, vecs = np.linalg.eigh(A)
    vals[vals<0] = 0
    lam_mat = np.diag(vals.tolist())
    A = vecs@lam_mat@vecs.T
    A = fractional_matrix_power(W,-0.5)@A@fractional_matrix_power(W,-0.5)

    return A

def higham(sigma,eps,maxiter):
    n = sigma.shape[0]
    di = np.diag_indices(n)
    error_0 = -np.inf
    delta_S = 0
    Y = sigma
    for k in range(maxiter):
        R = Y - delta_S
        X = eigen_con(R)
        delta_S = X-R
        Y = X.copy()
        Y[di] = 1
        error_1 = np.linalg.norm(Y-sigma)
        if error_1 - error_0 < eps:
            break
        else:
            error_0 = error_1
    return Y

# PCA simulation
def simulate_PCA(sigma,lam,nsim=25000):
    vals,vecs = np.linalg.eigh(sigma)
    idex = np.argsort(vals)[::-1]
    vals,vecs = vals[idex],vecs[:,idex]
    vals,vecs = vals[vals>0],vecs[:,vals>0]
    sum_vals = np.cumsum(vals)/sum(vals)
    vals,vecs = vals[sum_vals<=lam],vecs[:,sum_vals<=lam]
    np.random.seed(100)
    B = vecs@np.diag(vals.tolist())
    m = len(vals)
    r = np.random.normal(size=(m,nsim))

    return np.cov(B@r)