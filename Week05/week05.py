#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
#%%
def VaR_ES(sim_x,alpha=0.05):
    order_x = sorted(sim_x)
    n = alpha*len(order_x)
    up_n = int(np.ceil(n))
    dn_n = int(np.floor(n))
    VaR = -1*(order_x[up_n]+order_x[dn_n])/2
    ES = -1*np.mean(order_x[:dn_n+1])

    return VaR,ES

df = pd.read_csv('problem1.csv')
alpha = 0.05
# MLE for Normal Distribution
miu = df.mean()[0]
sigma = df.std(ddof=0)[0]
sim_norm = stats.norm.rvs(loc=miu,scale=sigma,size=10000,random_state=100)
VaR_norm,ES_norm = VaR_ES(sim_norm)

# MLE for t Distribution
def mle(parameters,x):
    miu = parameters[0]
    sigma = parameters[1]
    degree = parameters[2]
    L = np.sum(np.log(stats.t.pdf(x,loc=miu,scale=sigma,df = degree)))
    return -L
x = df.values.reshape(1,-1)
lik_model_t = minimize(mle,np.array([2,2,2]),args=(x,),method="SLSQP")
miu = lik_model_t.x[0]
sigma = lik_model_t.x[1]
df = lik_model_t.x[2]
sim_t = stats.t.rvs(loc=miu,scale=sigma,df = df,size=10000,random_state=100)
VaR_t,ES_t = VaR_ES(sim_t)

# plot histogram
# sns.histplot()
# sns.histplot()
# sns.histplot()
# %% Problem 3
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

portfolio = pd.read_csv('Portfolio.csv')
DailyPrices = pd.read_csv('DailyPrices.csv')
returns = return_calculate(DailyPrices).iloc[1:,]
currentprices = DailyPrices.iloc[-1,:]


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

def VaR_ES(order_x,alpha=0.05):
    n = alpha*order_x.shape[1]
    up_n = int(np.ceil(n))
    dn_n = int(np.floor(n))
    print(up_n,dn_n)
    VaR = (order_x[0,up_n]+order_x[0,dn_n])/2
    ES = np.mean(order_x[0,:dn_n+1])

    return VaR,ES

def portfolio_VaR_ES(portfolio,currentprices=currentprices,returns=returns,sim_n=1000):
    returns = returns.iloc[:,returns.columns.isin(portfolio['Stock'])]
    
    currentprices = currentprices[currentprices.index.isin(portfolio['Stock'])]
    holdings = [portfolio[portfolio.Stock==i]['Holding'] for i in returns.columns]
    holdings = np.array(holdings).reshape(-1,1)
    PV = currentprices.values@holdings

    rep_current = np.tile(currentprices.values,(sim_n,1))
    t_rvs = pd.DataFrame(columns=returns.columns)
    for i in returns.columns:
        ind_t_rvs = generalized_t_fitting(returns.loc[:,i].values,sim_n)
        t_rvs[i] = ind_t_rvs
    
    sim_prices = (t_rvs.values+1)*rep_current
    sim_prices = sim_prices@holdings
    sim_prices = sim_prices.T
    sim_prices.sort()
    VaR_price,ES_price = VaR_ES(sim_prices)


    return PV - ES_price, PV-VaR_price


ES_t = pd.DataFrame(columns=['A','B','C','Total'])
VaR_t= pd.DataFrame(columns=['A','B','C','Total'])

A = portfolio.loc[portfolio['Portfolio']=='A',]
ES_t['A'],VaR_t['A'] = portfolio_VaR_ES(A)

B = portfolio.loc[portfolio['Portfolio']=='B',]
ES_t['B'],VaR_t['B'] = portfolio_VaR_ES(B)

C = portfolio.loc[portfolio['Portfolio']=='C',]
ES_t['C'],VaR_t['C'] = portfolio_VaR_ES(C)

ES_t['Total'],VaR_t['Total']= portfolio_VaR_ES(portfolio)


# %%
