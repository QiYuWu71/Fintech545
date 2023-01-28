#%% Library importer
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import pacf,acf
from scipy.optimize import minimize
#%% Problem set 1
sample_size = 100
samples = 100
skew_list = np.zeros(samples)
kurt_list = np.zeros(samples)

for i in range(samples):
    x = np.random.normal(0,1,sample_size)
    skews = stats.skew(x)
    kurts = stats.kurtosis(x)
    skew_list[i] = skews
    kurt_list[i] = kurts

t_skew = (skew_list.mean()-0)/np.sqrt(skew_list.var()/samples)
p_skew = (1 - stats.t.cdf(abs(t_skew),df=sample_size-1))*2
t_kurt = (kurt_list.mean()-0)/np.sqrt(kurt_list.var()/samples)
p_kurt = (1 - stats.t.cdf(abs(t_kurt),df=sample_size-1))*2

#%% Problem set 2
dp2 = pd.read_csv("problem2.csv")
regr = linear_model.LinearRegression()
x = dp2.x.values.reshape(-1,1)
y = dp2.y.values.reshape(-1,1)
regr.fit(x,y)
y_pred = regr.predict(x)

sns.scatterplot(dp2,y='y',x='x')
plt.plot(x,y_pred,color = 'orange')
plt.title("Regression Fitting Plot")
plt.show()

error = y-y_pred
sm.qqplot(error,line="45")
plt.title("QQ Plot")
plt.show()
sns.histplot(error)
plt.title("Error Histogram Plot")
plt.show()
#%%
# MLE problem solving
# for normal distribution
def mle(parameters,x,y):
    m = parameters[0]
    b = parameters[1]
    sigma = parameters[2]
    y_exp = m*x+b
    L = np.sum(np.log(stats.norm.pdf(y-y_exp,loc=0,scale=sigma)))
    return -L

lik_model_normal = minimize(mle,np.array([2,2,2]),args=(x,y,))
m = lik_model_normal.x[0]
b = lik_model_normal.x[1]
sigma_norm = lik_model_normal.x[2]
y_pred_mle_norm = m*x+b

#%%
# for T distribution
def mle(parameters,x,y):
    m = parameters[0]
    b = parameters[1]
    sigma = parameters[2]
    degree = parameters[3]
    y_exp = m*x+b
    L = np.sum(np.log(stats.t.pdf(y-y_exp,loc=0,scale=sigma,df = degree)))
    return -L
lik_model_t = minimize(mle,np.array([2,2,2,2]),args=(x,y,))
m = lik_model_t.x[0]
b = lik_model_t.x[1]
sigma_t= lik_model_t.x[2]
degree = lik_model_t.x[3]
y_pred_mle_t= m*x+b
#%%
def AIC_BIC_norm(y,y_exp,sigma):
    k = 3
    L = np.sum(np.log(stats.norm.pdf(y-y_exp,loc=0,scale=sigma)))
    AIC = 2*k-2*L
    BIC = k*np.log(len(y))-2*L
    return AIC,BIC

def AIC_BIC_t(y,y_exp,sigma,degree):
    L = np.sum(np.log(stats.t.pdf(y-y_exp,loc=0,scale=sigma,df =degree)))
    k = 4
    AIC = 2*k-2*L
    BIC = k*np.log(len(y))-2*L
    return AIC,BIC

def R_sqaure(y,y_pred):
    miu = y_pred.mean()
    sst = np.dot((y-miu).T,y-miu)
    sse = np.dot((y-y_pred).T,y-y_pred)
    R_2 = 1 - sse/sst
    return R_2
#%%
AIC_norm,BIC_norm = AIC_BIC_norm(y,y_pred_mle_norm,sigma_norm)
R_square_norm = R_sqaure(y,y_pred_mle_norm)
AIC_t,BIC_t = AIC_BIC_t(y,y_pred,sigma_t,degree)
R_square_t = R_sqaure(y,y_pred_mle_t)
#%%
sns.scatterplot(dp2,y='y',x='x')
plt.plot(x,y_pred_mle_norm,color = 'orange',label='norm')
plt.plot(x,y_pred_mle_t,color="green",label='t')
plt.legend(loc=2)
plt.title("Regression Fitting Plot")
plt.show()
#%% Problem set 3
n = 1000
burn_in = 50
noise = np.random.normal(loc = 0,scale = 1,size=n+burn_in)
# AR1 AR2 AR3
# AR1: y_t = 1+0.5y_t-1 + e
y_AR1 = np.zeros(n)
y_last = 1
for i in range(1,n+burn_in):
    yt = 1 + 0.5*y_last + noise[i]
    y_last = yt
    if i >burn_in:
        y_AR1[i-burn_in] = yt

# AR2: y_t = 1+0.5y_t-1+0.5y_t-2
y_AR2 = np.zeros(n+burn_in)
y_AR2[0] = noise[0]
y_AR2[1] = noise[1]
for i in range(2,n+burn_in):
    y_AR2[i] = 1+0.5*y_AR2[i-1] - 0.5*y_AR2[i-2] + noise[i]
y_AR2 = y_AR2[burn_in:]

# AR3: y_t = 1 + 0.5y_t-1+0.5y_t-2 + 0.5y_t-3
y_AR3 = np.zeros(n+burn_in)
y_AR3[0] = noise[0]
y_AR3[1] = noise[1]
y_AR3[2] = noise[2]

for i in range(3,n+burn_in):
    y_AR3[i] = 1+0.5*y_AR3[i-1]-0.5*y_AR3[i-2]+0.5*y_AR3[i-3]+noise[i]
y_AR3 = y_AR3[burn_in:]

#%%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('AR Processes')
sns.lineplot(ax=axes[0],x=range(1000),y=y_AR1)
axes[0].set_title("AR(1)")
sns.lineplot(ax = axes[1],x = range(1000),y=y_AR2)
axes[1].set_title("AR(2)")
sns.lineplot(ax = axes[2],x = range(1000),y=y_AR3)
axes[2].set_title("AR(3)")

#%%
# MA1 MA2 MA3
y_MA1 = np.zeros(n+burn_in)
for i in range(1,n+burn_in):
    y_MA1[i] = noise[i] + 0.4*noise[i-1]
y_MA1 = y_MA1[burn_in:]

y_MA2 = np.zeros(n+burn_in)
for i in range(2,n+burn_in):
    y_MA2[i] = noise[i] + 0.4*noise[i-1]+ 0.5*noise[i-2]
y_MA2 = y_MA2[burn_in:]

y_MA3 = np.zeros(n+burn_in)
for i in range(3,n+burn_in):
    y_MA3[i] = noise[i] + 0.4*noise[i-1]+0.5*noise[i-2]+0.6*noise[i-3]
y_MA3 = y_MA3[burn_in:]
#%%
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('MA Processes')
sns.lineplot(ax=axes[0],x=range(1000),y=y_MA1)
axes[0].set_title("MA(1)")
sns.lineplot(ax = axes[1],x = range(1000),y=y_MA2)
axes[1].set_title("MA(2)")
sns.lineplot(ax = axes[2],x = range(1000),y=y_MA3)
axes[2].set_title("MA(3)")
#%%
# Autocovariance / Partial Autocovariance
# MA Process
n = len(acf(y_MA1))
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('MA Processes')
sns.lineplot(ax=axes[0],x=range(n),y=pacf(y_MA1),label="pacf")
sns.lineplot(ax=axes[0],x=range(n),y=acf(y_MA1),label="acf")
axes[0].set_title("MA(1)")
sns.lineplot(ax = axes[1],x = range(n),y=pacf(y_MA2),label="pacf")
sns.lineplot(ax = axes[1],x = range(n),y=acf(y_MA2),label="acf")
axes[1].set_title("MA(2)")
sns.lineplot(ax = axes[2],x = range(n),y=pacf(y_MA3),label="pacf")
sns.lineplot(ax = axes[2],x = range(n),y=acf(y_MA3),label="acf")
axes[2].set_title("MA(3)")
#%%
# AR Process
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('AR Processes')
sns.lineplot(ax=axes[0],x=range(n),y=pacf(y_AR1),label="pacf")
sns.lineplot(ax=axes[0],x=range(n),y=acf(y_AR1),label="acf")
axes[0].set_title("AR(1)")
sns.lineplot(ax = axes[1],x = range(n),y=pacf(y_AR2),label="pacf")
sns.lineplot(ax = axes[1],x = range(n),y=acf(y_AR2),label="acf")
axes[1].set_title("AR(2)")
sns.lineplot(ax = axes[2],x = range(n),y=pacf(y_AR3),label="pacf")
sns.lineplot(ax = axes[2],x = range(n),y=acf(y_AR3),label="acf")
axes[2].set_title("AR(3)")
# %%
