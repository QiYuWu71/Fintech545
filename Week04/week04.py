#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import statsmodels as sm
from mv_laplace import MvLaplaceSampler
#%% Question 1
rt = np.random.normal(0,2,90000000)
pt_1 = 5
## Classical Brownian Motion Pt = Pt-1 + rt
pt = pt_1+rt
print('The classical brownian motion std: {}'.format(np.std(pt)))
print('The classical brownian mean: {}'.format(np.mean(pt)))

## Arithmetic Return System
pt = pt_1*(1+rt)
print('The classical brownian motion std: {}'.format(np.std(pt)))
print('The classical brownian mean: {}'.format(np.mean(pt)))

## Log Return or Geometric Brownian Motion
pt = pt_1*np.exp(rt)
print('The Geometric brownian motion std: {}'.format(np.std(pt)))
print('The Geometric brownian mean: {}'.format(np.mean(pt)))

# %% Question 2 Var Calculation
# The return formula will be utilized to calculate return within the pandas dataframe
def return_calculate(prices,method='DISCRETE',dateColumn='date'):
    prices.set_index('Date',inplace=True)
    lst_prices = prices.shift(1)
    delta = prices/lst_prices

    if method.upper() == 'DISCRETE':
        delta = delta-1.0
    elif method.upper() =='LOG':
        delta = np.log(delta)
    else:
        raise ValueError('Error: {} does not exist', method)

    return delta
prices = pd.read_csv('DailyPrices.csv')

returns = return_calculate(prices)
# Meta:
Meta = returns.META
Meta = Meta - Meta.mean()
Meta = Meta.values[1:]
#%%
# Normal Distribution
alpha = 0.05
meta_scale = Meta.std()
VaR_norm = -1*stats.norm.ppf(alpha,loc=0,scale = meta_scale)
VaR_norm
#%%
# Normal Distribution with an Exponentially Weighted Variance
def gen_weight(lam,X):
    w = np.array([(1-lam)*(lam**i) for i in range(X.shape[0])])
    w_scale = w/sum(w)
    return w_scale

weight = gen_weight(0.94,Meta)[::-1]
meta_scale_ew = np.sqrt(np.sum(weight*((Meta)**2)))
VaR_norm_ew = -1*stats.norm.ppf(alpha,loc=0,scale = meta_scale_ew)
VaR_norm_ew
#%%
# T Distribution
def mle_t(parameters,x):
    sigma = parameters[0]
    degree = parameters[1]
    L = np.sum(np.log(stats.t.pdf(x,loc=0,scale=sigma,df = degree)))
    return -L
def sigma_cons(parameters):
    return parameters[0]
cons = {'type':'ineq','func':sigma_cons}
lik_model_t = minimize(mle_t,np.array([2,6]),method="SLSQP",args=(Meta,))
VaR_t = -1*stats.t.ppf(alpha,loc=0,scale=lik_model_t.x[0],df = lik_model_t.x[1])
VaR_t
#%%
# AR(1) Model
X = Meta[:len(Meta)-1].reshape(-1,1)
Y = Meta[1:].reshape(-1,1)
lrr = LinearRegression().fit(X,Y)
error = Y - lrr.predict(X)
error = error.reshape(1,-1)[0]
error.var()
VaR_AR = -1*stats.norm.ppf(alpha,loc=0,scale = error.std())
VaR_AR
#%%
# Historic Simulation
Npoints = np.random.randint(len(Meta),size = 100)
Meta_sim = Meta[Npoints]
Meta_sim.sort()
VaR_Hs = -1*np.percentile(Meta_sim,alpha*100)
VaR_Hs
#%% Question 3 VAR
portfolio = pd.read_csv('Portfolio.csv')
DailyPrices = pd.read_csv('DailyPrices.csv')
returns = return_calculate(DailyPrices).iloc[1:,]
currentprices = DailyPrices.iloc[-1,:]

def gen_weight(lam,X):
    w = np.array([(1-lam)*(lam**i) for i in range(X.shape[0])])[::-1]
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

def normalMC_VaR(portfolio,currentprices=currentprices,returns=returns,Ass_dist='Normal'):

    # initial data preprocessing
    returns = returns.iloc[:,returns.columns.isin(portfolio['Stock'])]
    currentprices = currentprices[currentprices.index.isin(portfolio['Stock'])]

    rep_current = np.tile(currentprices.values,(100000,1))
    weg_cov = weight_cov(0.94,returns.values)
    if Ass_dist == 'Normal':
        rs = np.random.multivariate_normal(returns.mean().values,weg_cov,size=100000)
    elif Ass_dist == 't':
        rs = stats.multivariate_t.rvs(returns.mean().values,weg_cov,df=100,size=100000,random_state=100)
    sim_prices = (1+rs)*rep_current
    holdings = [portfolio[portfolio.Stock==i]['Holding'] for i in returns.columns]
    holdings = np.array(holdings).reshape(-1,1)
    PV = currentprices.values@holdings
    
    sim_prices = sim_prices@holdings
    sim_prices.sort()
    sim_FPV = np.percentile(sim_prices,alpha*100)

    VaR = PV - sim_FPV

    return PV,VaR

PV_normal = pd.DataFrame(columns=['A','B','C','Total'])
VaR_normal = pd.DataFrame(columns=['A','B','C','Total'])

A = portfolio.loc[portfolio['Portfolio']=='A',]
PV_normal['A'],VaR_normal['A'] = normalMC_VaR(A)

B = portfolio.loc[portfolio['Portfolio']=='B',]
PV_normal['B'],VaR_normal['B'] = normalMC_VaR(B)

C = portfolio.loc[portfolio['Portfolio']=='C',]
PV_normal['C'],VaR_normal['C'] = normalMC_VaR(C)

PV_normal['Total'],VaR_normal['Total']= normalMC_VaR(portfolio)


np.random.seed(100)
# %%
VaR_t = pd.DataFrame(columns=['A','B','C','Total'])

A = portfolio.loc[portfolio['Portfolio']=='A',]
PV,VaR_t['A'] = normalMC_VaR(A,Ass_dist='t')

B = portfolio.loc[portfolio['Portfolio']=='B',]
PV,VaR_t['B'] = normalMC_VaR(B,Ass_dist='t')

C = portfolio.loc[portfolio['Portfolio']=='C',]
PV,VaR_t['C'] = normalMC_VaR(C,Ass_dist='t')

PV,VaR_t['Total']= normalMC_VaR(portfolio,Ass_dist='t')

# %%
