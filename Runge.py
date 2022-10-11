import numpy as np
import pandas as pd

KQ = np.array([[0,0,0,0],[0,0.5,0,0],[0,0,0,0],[0,0,0,0.55]])
TQ = np.array([[0],[0],[0],[0]])
Sig = np.array([[0.005,0,0,0],[0,0.01,0,0],[0,0,0.004,0],[0,0,0,0.015]])
H0 = Sig**2
L1 = np.array([[1], [2], [3], [4]])
L2 = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4],[1,2,3,4]])
KP = KQ-np.matmul(Sig,L2)
TPKPTrans = np.transpose(np.matmul(KQ,TQ)+np.matmul(Sig,L1))
rho1 = np.array([1,1,-1,-1])

def betamark(beta):
    betamark = -rho1+np.matmul(np.transpose(KP),beta)
    return betamark

def alphamark(beta):
    bT = np.transpose(beta)
    bTH0 = np.matmul(bT,H0)
    alphamark = np.matmul(TPKPTrans,beta)+(1/2)*np.matmul(bTH0, beta)
    return alphamark

def runge(endtime, beta0list, alpha0, h = 1/12):
    #Choose endtime = years, beta0list = [0,0,0,0] and alpha0=[0]
    # Returns a dict with keys as timestep ie 1/12, 2/12...
    #and first 4 values as betavalues and last as alphavalue
    n=(int)((endtime-beta0list[0])/h)
    betdict = {0: np.append(beta0list,alpha0)}
    for i in range(1,n+1):
        k1b = h * betamark(beta0list)
        k1a = h * alphamark(beta0list)
        k2b = h * betamark(beta0list+0.5*k1b)
        k2a = h * alphamark(beta0list+0.5*k1a)
        k3b = h * betamark(beta0list+0.5*k2b)
        k3a = h * alphamark(beta0list+0.5*k2a)
        k4b = h * betamark(beta0list+k3b)
        k4a = h * alphamark(beta0list+k3a)

        beta0list = beta0list + (1.0/6.0)*(k1b+2*k2b+2*k3b+k4b)
        alpha0 = alpha0 + (1.0/6.0)*(k1a+2*k2a+2*k3a+k4a)
        
        pair = np.append(beta0list, alpha0)

        betdict.update({h*i: pair})

    return betdict

ODE = runge(2, np.array([0,0,0,0]), 0)
print(ODE)
