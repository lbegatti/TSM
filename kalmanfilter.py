from typing import Any, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.linalg import expm

from nelsonSiegel import NelsonSiegel


class KalmanFilter(NelsonSiegel):
    def __init__(self, observedyield: pd.DataFrame, obs: int, timestep: float):
        self.implyields = None
        self.VbarUnc = None
        self.Vbar = None
        self.Sbar = None
        self.gainMatrix = None
        self.Sinv = None
        self.ith_loglikelihood = None
        self.Pt = None
        self.detS = None
        self.S = None
        self.res = None
        self.yt = None
        self.Pt_1 = None
        self.Xt_1 = None
        self.uncMean = None
        self.unconVar = None
        self.condVar = None
        self.Ct = None
        self.Ft = None
        self.Theta = None
        self.eigvecK = None
        self.eigvalK = None
        self.A = None
        self.Bmatrix = None
        self.H = None
        self.Sigma = None
        self.K = None
        self.sigma_err_sq = None
        self.lambda_R = None
        self.lambda_N = None
        self.sigma_RS = None
        self.sigma_RL = None
        self.sigma_NS = None
        self.sigma_NL = None
        self.theta4 = None
        self.theta3 = None
        self.theta2 = None
        self.theta1 = None
        self.kappa44 = None
        self.kappa43 = None
        self.kappa42 = None
        self.kappa41 = None
        self.kappa34 = None
        self.kappa33 = None
        self.kappa32 = None
        self.kappa31 = None
        self.kappa24 = None
        self.kappa23 = None
        self.kappa22 = None
        self.kappa21 = None
        self.kappa14 = None
        self.kappa13 = None
        self.kappa12 = None
        self.kappa11 = None
        self.obs = obs
        self.dt = timestep
        self.loglike = 0
        self.observedyield = observedyield

    def paramToOptimize(self, params):

        # reset the likelihood to zero after every iteration.
        self.loglike = 0

        para = params

        # initialize all the parameter values
        self.kappa11 = para[0]
        self.kappa12 = para[1]
        self.kappa13 = para[2]
        self.kappa14 = para[3]

        self.kappa21 = para[4]
        self.kappa22 = para[5]
        self.kappa23 = para[6]
        self.kappa24 = para[7]

        self.kappa31 = para[8]
        self.kappa32 = para[9]
        self.kappa33 = para[10]
        self.kappa34 = para[11]

        self.kappa41 = para[12]
        self.kappa42 = para[13]
        self.kappa43 = para[14]
        self.kappa44 = para[15]

        self.theta1 = para[16]
        self.theta2 = para[17]
        self.theta3 = para[18]
        self.theta4 = para[19]

        # Force positive
        self.sigma_NL = abs(para[20])
        self.sigma_NS = abs(para[21])
        self.sigma_RL = abs(para[22])
        self.sigma_RS = abs(para[23])
        self.lambda_N = para[24]
        self.lambda_R = para[25]
        self.sigma_err_sq = para[26] ** 2

        self.K = np.array([self.kappa11, self.kappa12, self.kappa13, self.kappa14,
                           self.kappa21, self.kappa22, self.kappa23, self.kappa24,
                           self.kappa31, self.kappa32, self.kappa33, self.kappa34,
                           self.kappa41, self.kappa42, self.kappa43, self.kappa44]).reshape(4, 4)

        self.Theta = np.array([self.theta1, self.theta2, self.theta3, self.theta4])

        self.Sigma = np.diag([self.sigma_NL, self.sigma_NS, self.sigma_RL, self.sigma_RS])

        self.H = np.diag([self.sigma_err_sq for _ in range(10)])

    def checkEigen(self):
        self.eigvalK, self.eigvecK = np.linalg.eig(self.K)

        if all(self.eigvalK.real > 0):

            return True

        else:

            # print('At least one eigenvalue is negative or complex!') #prints too much
            return False

    def calcAB(self):

        nelsonSiegelN = NelsonSiegel(Lambda=self.lambda_N, sigmaS=self.sigma_NS, sigmaL=self.sigma_NL,
                                     Xt=self.Theta[0:2],  # only the two first
                                     tauList=[2, 3, 5, 7, 10])
        nelsonSiegelR = NelsonSiegel(Lambda=self.lambda_R, sigmaL=self.sigma_RL, sigmaS=self.sigma_RS,
                                     Xt=self.Theta[2:],  # only the two last
                                     tauList=[2, 3, 5, 7, 10])
        ANom = nelsonSiegelN.Avector()
        AReal = nelsonSiegelR.Avector()
        self.A = np.array([ANom, AReal]).ravel()
        self.Bmatrix = np.zeros(40).reshape(10, 4)
        self.Bmatrix[0:5, 0:2] = nelsonSiegelN.Bmatrix()
        self.Bmatrix[5:10, 2:4] = nelsonSiegelR.Bmatrix()

    def condUncCov(self):

        # Cond and uncond covariance matrix

        ## S-overline matrix
        self.Sbar = (np.linalg.inv(self.eigvecK) @ self.Sigma @ self.Sigma.T @ np.linalg.inv(self.eigvecK).T).real  #

        ## Step 2: V-overline
        self.Vbar = np.zeros((4, 4), dtype=complex)
        for i in range(0, len(self.Sbar)):
            for j in range(0, len(self.Sbar)):
                self.Vbar[i, j] = (self.Sbar[i, j] / (self.eigvalK[i] + self.eigvalK[j])) * (
                        1 - np.exp(-(self.eigvalK[i] + self.eigvalK[j]) * self.dt))

        self.VbarUnc = np.zeros((4, 4), dtype=complex)
        for i in range(0, len(self.Sbar)):
            for j in range(0, len(self.Sbar)):
                self.VbarUnc[i, j] = (self.Sbar[i, j] / (self.eigvalK[i] + self.eigvalK[j]))

        self.condVar = (self.eigvecK @ self.Vbar @ self.eigvecK.T).real  # QVbar_dtQT
        self.unconVar = (self.eigvecK @ self.VbarUnc @ self.eigvecK.T).real  # QVbar_infQT
        self.uncMean = self.Theta

    def calcFC(self):

        self.Ft = expm(-self.K * self.dt)
        self.Ct = (np.eye(4) - self.Ft) @ self.Theta

    def kalmanfilter(self, pars):

        self.paramToOptimize(params=pars)
        if not self.checkEigen():
            return 999999
        self.calcAB()
        self.condUncCov()
        self.calcFC()

        self.Xt = self.Theta
        self.Pt = self.unconVar
        for o in range(self.obs):

            # Prediction step
            self.Xt_1 = self.Ft @ self.Xt + self.Ct
            self.Pt_1 = self.Ft @ self.Pt @ self.Ft.T + self.condVar

            # Observed implied yields
            self.yt = np.array(self.observedyield.iloc[o].values)

            # Residuals
            self.res = self.yt - self.A - self.Bmatrix @ self.Xt_1

            # Cov matrix prefit residuals
            self.S = self.H + self.Bmatrix @ self.Pt_1 @ self.Bmatrix.T

            # Det of cov matrix prefit residuals
            self.detS = np.linalg.det(self.S)

            if self.detS > 0:
                # Inverse of cov matrix
                self.Sinv = np.linalg.inv(self.S)

                # Kalman gain matrix
                self.gainMatrix = self.Pt_1 @ self.Bmatrix.T @ self.Sinv

                # Updated step
                self.Xt = self.Xt_1 + self.gainMatrix @ self.res
                self.Pt = self.Pt_1 - self.gainMatrix @ self.Bmatrix @ self.Pt_1

                # Log likelihood for each obs step
                self.loglike += - 0.5 * (np.log(self.detS) + self.res.T @ self.Sinv @ self.res)

            else:

                # print('Determinant is not-positive.') #prints to much
                return 888888
        print(f'                                     loglike: {-self.loglike}', end='\r')
        return -self.loglike

    def kalmanFilterFinal(self, pars) -> Union[int, tuple[DataFrame, Any, Any, Any, DataFrame]]:

        self.paramToOptimize(params=pars)
        self.checkEigen()
        self.calcAB()
        self.condUncCov()
        self.calcFC()

        self.Xt = self.Theta
        self.Pt = self.unconVar
        impliedYield_final = []
        finalXdata = [self.Xt]

        for o in range(self.obs):

            # Prediction step
            self.Xt_1 = self.Ft @ self.Xt + self.Ct
            self.Pt_1 = self.Ft @ self.Pt @ self.Ft.T + self.condVar

            # Observed implied yields
            self.yt = np.array(self.observedyield.iloc[o].values)

            # Residuals
            self.res = self.yt - self.A - self.Bmatrix @ self.Xt_1

            # Cov matrix prefit residuals
            self.S = self.H + self.Bmatrix @ self.Pt_1 @ self.Bmatrix.T

            # Det of cov matrix prefit residuals
            self.detS = np.linalg.det(self.S)

            if self.detS > 0:

                # Inverse of cov matrix
                self.Sinv = np.linalg.inv(self.S)

                # Kalman gain matrix
                self.gainMatrix = self.Pt_1 @ self.Bmatrix.T @ self.Sinv

                # Updated step
                self.Xt = self.Xt_1 + self.gainMatrix @ self.res
                self.Pt = np.eye(4) @ self.Pt_1 - self.gainMatrix @ self.Bmatrix @ self.Pt_1
                finalXdata.append(self.Xt)

                ## Model implied yields
                self.implyields = self.A + self.Bmatrix @ self.Xt
                impliedYield_final.append(self.implyields)

                # log likelihood for each obs step

                self.loglike += - 0.5 * (np.log(self.detS) + self.res.T @ self.Sinv @ self.res)

            else:

                # print('Determinant is not-positive.') #prints to much
                return 888888

        impliedYield_final_df = pd.DataFrame(impliedYield_final)
        finalXdata = pd.DataFrame(finalXdata)
        return impliedYield_final_df, self.K, self.Theta, self.Sigma, finalXdata
