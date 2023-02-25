from scipy import stats
from scipy import stats
from scipy.optimize import minimize
import numpy as np
from sklearn.linear_model import LinearRegression

def normal_VaR(data,alpha=0.05):
    meta_scale = data.std()
    VaR_norm = -1*stats.norm.ppf(alpha,loc=0,scale = meta_scale)
    return VaR_norm

def gen_weight(lam,X):
    w = np.array([(1-lam)*(lam**i) for i in range(X.shape[0])])
    w_scale = w/sum(w)
    return w_scale

def normal_VaR_Ew(lam,X,alpha=0.05):
    weight = gen_weight(lam,X)[::-1]
    meta_scale_ew = np.sqrt(np.sum(weight*((X)**2)))
    VaR_norm_ew = -1*stats.norm.ppf(alpha,loc=0,scale = meta_scale_ew)
    return VaR_norm_ew

def mle_t(parameters,x):
    miu = parameters[0]
    sigma = parameters[1]
    degree = parameters[2]
    L = np.sum(np.log(stats.t.pdf(x,loc=miu,scale=sigma,df = degree)))
    return -L

def t_VaR(X,mle_t=mle_t,alpha = 0.05):
    lik_model_t = minimize(mle_t,np.array([2,2,6]),method="SLSQP",args=(X,))
    VaR_t = -1*stats.t.ppf(alpha,loc=lik_model_t.x[0],scale=lik_model_t.x[1],df = lik_model_t.x[2])
    return VaR_t

def AR_VaR(X,Y,alpha=0.05):
    lrr = LinearRegression().fit(X,Y)
    error = Y - lrr.predict(X)
    error = error.reshape(1,-1)[0]
    error.var()
    VaR_AR = -1*stats.norm.ppf(alpha,loc=0,scale = error.std())
    return VaR_AR

def His_VaR(X,alpha=0.05):
    Npoints = np.random.randint(len(X),size = 100)
    Meta_sim = X[Npoints]
    Meta_sim.sort()
    VaR_Hs = -1*np.percentile(Meta_sim,alpha*100)
    return VaR_Hs
