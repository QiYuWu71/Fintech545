import numpy as np

def AR1(n,burn_in,noise,beta1):
    np.random.seed(100)
    noise = np.random.normal(loc = 0,scale = 1,size=n+burn_in)
    y_AR1 = np.zeros(n)
    y_last = 1
    for i in range(1,n+burn_in):
        yt = 1 + beta1*y_last + noise[i]
        y_last = yt
        if i >burn_in:
            y_AR1[i-burn_in] = yt
    return y_AR1

def AR2(n,burn_in,noise,beta1,beta2):
    np.random.seed(100)
    noise = np.random.normal(loc = 0,scale = 1,size=n+burn_in)
    y_AR2 = np.zeros(n+burn_in)
    y_AR2[0] = noise[0]
    y_AR2[1] = noise[1]
    for i in range(2,n+burn_in):
        y_AR2[i] = 1+beta1*y_AR2[i-1] +beta2*y_AR2[i-2] + noise[i]
    y_AR2 = y_AR2[burn_in:]
    return y_AR2

def AR3(n,burn_in,noise,beta1,beta2,beta3):
    np.random.seed(100)
    noise = np.random.normal(loc = 0,scale = 1,size=n+burn_in)
    y_AR3 = np.zeros(n+burn_in)
    y_AR3[0] = noise[0]
    y_AR3[1] = noise[1]
    y_AR3[2] = noise[2]

    for i in range(3,n+burn_in):
        y_AR3[i] = 1+beta1*y_AR3[i-1]+beta2*y_AR3[i-2]+beta3*y_AR3[i-3]+noise[i]
    y_AR3 = y_AR3[burn_in:]
    return y_AR3

def MA1(n,burn_in,noise,beta1):
    np.random.seed(100)
    noise = np.random.normal(loc = 0,scale = 1,size=n+burn_in)
    y_MA1 = np.zeros(n+burn_in)
    for i in range(1,n+burn_in):
        y_MA1[i] = noise[i] + beta1*noise[i-1]
    y_MA1 = y_MA1[burn_in:]
    return y_MA1

def MA2(n,burn_in,noise,beta1,beta2):
    np.random.seed(100)
    noise = np.random.normal(loc = 0,scale = 1,size=n+burn_in)
    y_MA2 = np.zeros(n+burn_in)
    for i in range(2,n+burn_in):
        y_MA2[i] = noise[i] + beta1*noise[i-1]+ beta2*noise[i-2]
    y_MA2 = y_MA2[burn_in:]
    return

def MA3(n,burn_in,noise,beta1,beta2,beta3):
    np.random.seed(100)
    noise = np.random.normal(loc = 0,scale = 1,size=n+burn_in)

    y_MA3 = np.zeros(n+burn_in)
    for i in range(3,n+burn_in):
        y_MA3[i] = noise[i] + beta1*noise[i-1]+beta2*noise[i-2]+beta3*noise[i-3]
    y_MA3 = y_MA3[burn_in:]

    return y_MA3