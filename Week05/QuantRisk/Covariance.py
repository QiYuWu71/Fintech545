#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
def gen_weight(lam,X):
    w = np.array([(1-lam)*(lam**i) for i in range(X.shape[0])])
    w_scale = w/sum(w)
    return w_scale

def weight_cov(lam,X):
    cov = np.zeros((X.shape[1],X.shape[1]))
    w = gen_weight(lam,X)
    for i in range(X.shape[1]):
        for j in range(i,X.shape[1]):
            if i == j:
                m = X[:,i]
                scale_m = (m-m.mean())**2
                cov[i,j] = np.sum(w*scale_m)
            if i!= j:
                m = X[:,i]
                n = X[:,j]
                scale_m = (m-m.mean())
                scale_n = (n-n.mean())
                cov[i,j] = np.sum(w*scale_m*scale_n)
                cov[j,i] = cov[i,j]
    return cov

def EWvar_EWcorr(X,lam=0.97):
    cov = np.zeros((X.shape[1],X.shape[1]))
    w = gen_weight(lam,X)
    for i in range(X.shape[1]):
        for j in range(i,X.shape[1]):
            if i == j:
                m = X[:,i]
                scale_m = (m-m.mean())**2
                cov[i,j] = np.sum(w*scale_m)
            if i!= j:
                m = X[:,i]
                n = X[:,j]
                scale_m = (m-m.mean())
                scale_n = (n-n.mean())
                cov[i,j] = np.sum(w*scale_m*scale_n)
                cov[j,i] = cov[i,j]
    return cov

# Pearson var + corr
def Pvar_Pcorr(X):
    corr = np.corrcoef(X.T)
    var_lst = np.sqrt(np.var(X,axis=0))
    var_mat = np.diag(var_lst.tolist())
    cov = var_mat@corr@var_mat
    return cov

# EW var + Pearson corr
def EWvar_Pcorr(X,lam):
    n = X.shape[1]
    corr = np.corrcoef(X.T)
    w = gen_weight(lam,X)
    var_mat = np.zeros((n,n))
    for i in range(n):
        m = X[:,i]
        scale_m = (m-m.mean())**2
        var_mat[i,i] = np.sqrt(np.sum(w*scale_m))
    cov = var_mat@corr@var_mat
    return cov

# Pearson var + EW corr
def EWcorr(X,lam=0.97):
    n = X.shape[1]
    corr = np.zeros((n,n))
    w = gen_weight(lam,X)
    for i in range(n):
        m = X[:,i]
        scale_m = (m-m.mean())**2
        corr[i,i] = np.sum(w*scale_m)
    for i in range(n):
        for j in range(i,n):
            m = X[:,i]
            k = X[:,j]
            cov = np.sum(w*(m-m.mean())*(k-k.mean()))
            corr[i,j] = cov/np.sqrt(corr[i,i]*corr[j,j])
            corr[j,i] = corr[i,j]
    return corr

def Pvar_EWcorr(X,lam=0.97):
    n = X.shape[0]
    var_lst = np.sqrt(np.var(X,axis=0))
    var_mat = np.diag(var_lst.tolist())
    corr = EWcorr(X,lam)
    cov = var_mat@corr@var_mat
    return cov