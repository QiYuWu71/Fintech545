#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
import time
#%% Problem 1. Exponentially Weighted Covariance Matrix
dailyR = pd.read_csv('DailyReturn.csv')
dailyR = dailyR.rename(columns = {'Unnamed: 0':'Date'})
dailyR.sort_index(ascending=False,inplace=True)
dailyR.set_index('Date',inplace=True)
X = dailyR.values

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

def gen_pcaplot(lam,X=X):
    cov = weight_cov(lam,X)
    val = np.linalg.eigvals(cov)
    val = sorted(val,reverse=True)
    val = [i for i in val if i>0]
    val = np.cumsum(val)/sum(val)
    plt.plot(range(len(val)),val,label = str(round(lam,2)))


for lam in np.linspace(0.1,0.95,10):
    gen_pcaplot(lam)
plt.title('The PCA Explanation')
plt.legend()
plt.show()

#%% Problem 2. Cholesky Decomposition & Near Positive Semidefinite
# Cholesky PSD matrix
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
#%% generate data
n = 500
sigma = np.full((n,n),0.9)
for i in range(n):
    sigma[i,i] = 1
sigma[0,1] = 0.7357
sigma[1,0] = 0.7357
#%%
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
#%%
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
#%%
import seaborn as sn
n = 2000
sigma = np.full((n,n),0.9)
for i in range(n):
    sigma[i,i] = 1
    sigma[0,1] = 0.7357
    sigma[1,0] = 0.7357
start = time.time()
npsd = near_psd(sigma,0.0)
psd_time = time.time()-start
psd_norm = np.linalg.norm(npsd-sigma)
vals = np.linalg.eigvals(npsd)
print('The negative eigen value:',vals[vals<0])

start = time.time()
nhigh = higham(sigma,0.00001,1000)
high_time = time.time()-start
high_norm = np.linalg.norm(nhigh-sigma)

vals = np.linalg.eigvals(nhigh)
print('The negative eigen value:',vals[vals<0])

run_time = [psd_time,high_time]
diff_norm = [psd_norm,high_norm]

#%%
plt.bar(['Near_PSD','Higham'],diff_norm,color = ['blue','red'])
plt.title('N = '+str(n)+' Distance')
plt.show()
plt.bar(['Near_PSD','Highham'],run_time,color = ['blue','red'])
plt.title('N = '+str(n)+' Run Time')
plt.show()


#%% Problem 3. Principle Component Analysis
# read the csv file, in order to have the best 
dailyR = pd.read_csv('DailyReturn.csv')
dailyR = dailyR.rename(columns = {'Unnamed: 0':'Date'})
dailyR.sort_index(ascending=False,inplace=True)
dailyR.set_index('Date',inplace=True)
X = dailyR.values
#%%
# EW variance + EW correlation
def gen_weight(X,lam=0.97):
    w = np.array([(1-lam)*(lam**i) for i in range(X.shape[0])])
    w_scale = w/sum(w)
    return w_scale

def EWvar_EWcorr(X,lam=0.97):
    cov = np.zeros((X.shape[1],X.shape[1]))
    w = gen_weight(X,lam)
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
def EWvar_Pcorr(X):
    n = X.shape[1]
    corr = np.corrcoef(X.T)
    w = gen_weight(X)
    var_mat = np.zeros((n,n))
    for i in range(n):
        m = X[:,i]
        scale_m = (m-m.mean())**2
        var_mat[i,i] = np.sqrt(np.sum(w*scale_m))
    print("corr_mat\n",corr)
    cov = var_mat@corr@var_mat
    return cov

# Pearson var + EW corr
def EWcorr(X,lam=0.97):
    n = X.shape[1]
    corr = np.zeros((n,n))
    w = gen_weight(X)
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

def Pvar_EWcorr(X):
    n = X.shape[0]
    var_lst = np.sqrt(np.var(X,axis=0))
    var_mat = np.diag(var_lst.tolist())
    corr = EWcorr(X)
    cov = var_mat@corr@var_mat
    return cov
#%%
def simulate_PCA(sigma,lam,nsim=25000):
    vals,vecs = np.linalg.eig(sigma)
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
#%%
# EW variance + EW correlation
EE_var= EWvar_EWcorr(X)
# Pearson var + corr
PP_var= Pvar_Pcorr(X)
# EW var + Pearson corr
EP_var = EWvar_Pcorr(X)
# Pearson var + EW corr
PE_var = Pvar_EWcorr(X)
# %%
def generate_pca(sigma):
    runinfo = []
    starttime = time.time()
    Xdata = np.random.multivariate_normal(mean = np.zeros(101),cov=sigma,size=25000)
    X_cov_dir = np.cov(Xdata.T)
    run = round(time.time()-starttime,5)
    runinfo.append(run)

    starttime = time.time()
    X_cov_pca100 = simulate_PCA(sigma,1)
    run = round(time.time()-starttime,5)
    runinfo.append(run)

    starttime = time.time()
    X_cov_pca75 = simulate_PCA(sigma,0.75)
    run = round(time.time()-starttime,5)
    runinfo.append(run)

    starttime = time.time()
    X_cov_pca50 = simulate_PCA(sigma,0.5)
    run = round(time.time()-starttime,5)
    runinfo.append(run)

    return X_cov_dir,X_cov_pca100,X_cov_pca75,X_cov_pca50,runinfo

def generate_dis(sigma):
    origi = np.cov(X.T)
    X_cov_dir,X_cov_pca100,X_cov_pca75,X_cov_pca50,runinfo = generate_pca(sigma)
    direct = np.linalg.norm(X_cov_dir-origi)
    pca100 = np.linalg.norm(X_cov_pca100-origi)
    pca75 = np.linalg.norm(X_cov_pca75-origi)
    pca50 = np.linalg.norm(X_cov_pca50-origi)
    dist = [direct,pca100,pca75,pca50]
    return dist,runinfo

# EE covariance
dist,runinfo = generate_dis(EE_var)
print('EE',dist)
plt.bar(['Director','PCA100%','PCA75%','PCA50%'],dist,color = ['red','orange','blue','green'])
plt.title('EW variance + EW correlation Distance')
plt.show()
plt.bar(['Director','PCA100%','PCA75%','PCA50%'],runinfo,color = ['red','orange','blue','green'])
plt.title('EW variance + EW correlation Run Time')
plt.show()
#%%
# Pearson covariance
dist,runinfo = generate_dis(PP_var)
print('PP',dist)
plt.bar(['Director','PCA100%','PCA75%','PCA50%'],dist,color = ['red','orange','blue','green'])
plt.title('Pearson var + corr Distance')
plt.show()
plt.bar(['Director','PCA100%','PCA75%','PCA50%'],runinfo,color = ['red','orange','blue','green'])
plt.title('Pearson var + corr Run Time')
plt.show()
# EP covariance
dist,runinfo = generate_dis(EP_var)
print('EP',dist)
plt.bar(['Director','PCA100%','PCA75%','PCA50%'],dist,color = ['red','orange','blue','green'])
plt.title('EW var + Pearson corr Distance')
plt.show()
plt.bar(['Director','PCA100%','PCA75%','PCA50%'],runinfo,color = ['red','orange','blue','green'])
plt.title('EW var + Pearson corr Run Time')
plt.show()
# PE covariance
dist,runinfo = generate_dis(PE_var)
print('PE',dist)
plt.bar(['Director','PCA100%','PCA75%','PCA50%'],dist,color = ['red','orange','blue','green'])
plt.title('Pearson var + EW corr Distance')
plt.show()
plt.bar(['Director','PCA100%','PCA75%','PCA50%'],runinfo,color = ['red','orange','blue','green'])
plt.title('Pearson var + EW corr Run Time')
plt.show()
# %%
