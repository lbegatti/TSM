import numpy as np
import pandas as pd

# for i in range(len(tenors)):





TP = finalTheta
Sig = finalSigma
H0 = Sig @ Sig.T
KP = finalK
TPKPTrans = (KP @ TP).T
rho1 = np.array([1,1,-1,-1])

def betamark(beta):
    betamark = -rho1+np.matmul(np.transpose(KP),beta)
    return betamark

def alphamark(beta):
    bT = np.transpose(beta)
    bTH0 = np.matmul(bT,H0)
    alphamark = np.matmul(TPKPTrans,beta)+(1/2)*np.matmul(bTH0, beta)
    return alphamark

def runge(tau, beta0list, alpha0, h, Xt):
    #Choose endtime = years, beta0list = [0,0,0,0] and alpha0=[0]
    # Returns a dict with keys as timestep ie 1/12, 2/12...
    #and first 4 values as betavalues and last as alphavalue
    n=int(tau/h)
    betdict = {0: np.append(beta0list,alpha0)}
    allbetas = []
    allalphas = []
    for i in range(1,n):
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
        #allbetas.append(beta0list)
        #allalphas.append(alpha0)
        

    return np.exp(alpha0 - np.transpose(beta0list) @ Xt)





inf2 = []
for i in range(len(finalXdata)):
    inflation = -np.log(runge(tau = 10, beta0list = [0,0,0,0], alpha0 = 0, h = 1/100, Xt = finalXdata.iloc[i].values))/10
    inf2.append(inflation)

print(inf2)

'''
def mod_imp_inf(tau):
    list_of_imp_inf = []
    for i in range(len(alphavalues)):
        result = (alphavalues[i]+np.matmul(np.transpose(betavalues[i]),Xt))/(-tau)
        list_of_imp_inf.append(result)
    return list_of_imp_inf
'''