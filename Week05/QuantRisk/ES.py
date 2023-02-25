import numpy as np
from scipy import stats
from scipy.optimize import minimize

def VaR_ES(sim_x,alpha=0.05):
    order_x = sorted(sim_x)
    n = alpha*len(order_x)
    up_n = int(np.ceil(n))
    dn_n = int(np.floor(n))
    VaR = -1*(order_x[up_n]+order_x[dn_n])/2
    ES = -1*np.mean(order_x[:dn_n+1])

    return VaR,ES

def mle_t(parameters,x):
    miu = parameters[0]
    sigma = parameters[1]
    degree = parameters[2]
    L = np.sum(np.log(stats.t.pdf(x,loc=miu,scale=sigma,df = degree)))
    return -L

def generalized_t_fitting(rt,sim_n):
    lik_model_t = minimize(mle_t,np.array([1,1,1]),args=(rt,),method="SLSQP")
    miu = lik_model_t.x[0]
    sigma = lik_model_t.x[1]
    df = lik_model_t.x[2]
    sim_t = stats.t.rvs(loc=miu,scale=sigma,df = df,size=sim_n,random_state=100)
    return sim_t
