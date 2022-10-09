import numpy as np
import pandas as pd
from nelsonSiegel import NelsonSiegel


def kalman(para:np.array, dt:float, steps:int, obs_yields:pd.DataFrame, n_obs:int=10):
    kappa11=para[0]
    kappa12=para[1]
    kappa13=para[2]
    kappa14=para[3]

    kappa21=para[4]
    kappa22=para[5]
    kappa23=para[6]
    kappa24=para[7]

    kappa31=para[8]
    kappa32=para[9]
    kappa33=para[10]
    kappa34=para[11]

    kappa41=para[12]
    kappa42=para[13]
    kappa43=para[14]
    kappa44=para[15]

    theta1=para[16]
    theta2=para[17]
    theta3=para[18]
    theta4=para[19]

    sigma_NL=abs(para[20])
    sigma_NS=abs(para[21])
    sigma_RL=abs(para[22])
    sigma_RS=abs(para[23])

    lambda_N=para[24]
    lambda_R=para[25]

    sigma_err_sq=np.square(para[26])

    K=np.array([kappa11,kappa12,kappa13,kappa14,
                kappa21,kappa22,kappa23,kappa24,
                kappa31,kappa32,kappa33,kappa34,
                kappa41,kappa42,kappa43,kappa44]).reshape(4,4)

    theta=np.array([theta1,theta2,theta3,theta4])

    Sigma=np.diag([sigma_NL,sigma_NS,sigma_RL,sigma_RS])

    H=np.diag(np.repeat(sigma_err_sq, n_obs))

    #Calculating eigen values of K:
    phi, Q=np.linalg.eig(K)
    phi, Q=phi.real, Q.real
    #if (Keig[0]<=0).any() or np.iscomplex(Keig[0]).any():
    if (phi<=0).any():
        return 9999999


    #Calculating A and B
    maturities=[2,3,5,7,10]
    nelsonNominal=NelsonSiegel(Lambda=lambda_N, sigmaL=sigma_NL, sigmaS=sigma_NS, Xt=theta[0:2],tauList=maturities)
    nelsonReal=NelsonSiegel(Lambda=lambda_R, sigmaL=sigma_RL, sigmaS=sigma_RS, Xt=theta[2:4],tauList=maturities)
    
    A=np.array([nelsonNominal.Avector(),nelsonReal.Avector()]).reshape(n_obs)
    B=np.zeros(40).reshape(10,4)
    B[0:5,0:2]=nelsonNominal.Bmatrix()
    B[5:,2:]=nelsonReal.Bmatrix()

    #Calculating Sbar
    Sbar=np.dot(np.dot(np.dot(np.linalg.pinv(Q),Sigma),Sigma.T), np.linalg.pinv(Q).T).real
    
    #Calculating Vbar for the timestep
    Vbardt=np.zeros(16).reshape(4,4).real
    Vbarinf=np.zeros(16).reshape(4,4).real
    for i in range(4):
        for j in range(4):
            frac=Sbar[i,j]/(phi[i]+phi[j])
            Vbardt[i,j]=frac*(1-np.exp(-(phi[i]+phi[j])*dt))
            Vbarinf[i,j]=frac
    
    #ConditionalCovariance
    cCov=np.dot(Q,np.dot(Vbardt,Q.T)).real

    #UnconditonalCovariance
    uCov=np.dot(Q,np.dot(Vbarinf,Q.T)).real

    #Start the filter
    X=theta #"conditional X based on measurement"
    P=uCov

    #Calculate F_t and C_t
    Ct=np.dot(theta, np.eye(4)-np.exp(-K*dt)).real
    Ft=np.exp(-K*dt).real

    #Initializing loglikelihood
    loglike=0
    #Iterate over all the observation dates
    i=0
    while i<steps:
        print(i)
        #Prediction step
        X=Ct+np.dot(Ft,X)
        P=np.dot(Ft, np.dot(P,Ft.T))+cCov
        print(P)
        #calculate the model-implied yields
        y=-A+np.dot(-B,X)

        #Calculate the prefit residuals based on the observed and implied
        residual=np.array(obs_yields.iloc[i])-y #called nu in notes

        #Calculate the covariance matrix of the prefit residuals
        S=H+np.dot(B, np.dot(P, B.T))
        detS=np.linalg.det(S)
        if (detS<=0):
            return 8888888

        logdetS=np.log(detS)

        #Inverse of the covariance
        Sinv=np.linalg.pinv(S)

        #Calculating the Kalman gain matrix:
        KalmanGM=np.dot(P,np.dot(B.T, Sinv))

        #update step:
        X=X+np.dot(KalmanGM, residual)
        P=np.dot(np.eye(4)-np.dot(KalmanGM,B),P)
        print(P)    
        #Calculating the likelihood
        loglike+= -0.5*(logdetS+np.dot(residual.T,np.dot(Sinv,residual)))
        i+=1
    
    return -loglike

