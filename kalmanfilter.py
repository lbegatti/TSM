from nelsonSiegel import NelsonSiegel
import pandas as pd
import numpy as np
from scipy import optimize


class KalmanFilter(NelsonSiegel):
    def __init__(self, parameters, observedyield: pd.DataFrame, obs: int, timestep: float):
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
        self.para = parameters
        self.obs = obs
        self.dt = timestep
        self.loglike = 0
        self.observedyield = observedyield

    def checkEigen(self):
        self.eigvalK, self.eigvecK = np.linalg.eig(self.K)
        if self.eigvalK.all() > 0:
            pass
        else:
            print('problem')

    def calcAB(self):
        nelsonSiegelN = NelsonSiegel(Lambda=self.lambda_N, sigmaS=self.sigma_NS, sigmaL=self.sigma_NL, Xt=self.Theta,
                                     tauList=[2, 3, 5, 7, 10])
        nelsonSiegelR = NelsonSiegel(Lambda=self.lambda_R, sigmaL=self.sigma_RL, sigmaS=self.sigma_RS, Xt=self.Theta,
                                     tauList=[2, 3, 5, 7, 10])
        ANom = nelsonSiegelN.Avector()
        AReal = nelsonSiegelR.Avector()
        self.A = np.array([ANom, AReal]).reshape(1, 10)
        self.Bmatrix = np.zeros(40).reshape(10, 4)
        self.Bmatrix[0:5, 0:2] = nelsonSiegelN.Bmatrix()
        self.Bmatrix[5:10, 2:4] = nelsonSiegelR.Bmatrix()

    def condUncCov(self):

        # Cond and uncond covariance matrix
        ## S-overline matrix
        Sbar = np.dot(np.dot(np.linalg.inv(self.eigvecK), np.dot(self.Sigma, self.Sigma.T)),
                      np.linalg.inv(self.eigvecK).T)
        Sbar = Sbar.real

        ## Step 2: V-overline
        Vbar = np.zeros(16).reshape(Sbar.shape)
        for i in range(0, len(Sbar)):
            for j in range(0, len(Sbar)):
                Vbar[i, j] = (Sbar[i, j] / (self.eigvalK[i] + self.eigvalK[j])) * \
                             (1 - np.exp(-(self.eigvalK[i] + self.eigvalK[j]) * self.dt))

        VbarUnc = np.zeros(16).reshape(Sbar.shape)
        for i in range(0, len(Sbar)):
            for j in range(0, len(Sbar)):
                VbarUnc[i, j] = (Sbar[i, j] / (self.eigvalK[i] + self.eigvalK[j]))

        # take the real part only
        Vbar = Vbar.real
        VbarUnc = VbarUnc.real

        self.condVar = np.dot(self.eigvecK, np.dot(Vbar, self.eigvecK.T))
        self.unconVar = np.dot(self.eigvecK, np.dot(VbarUnc, self.eigvecK.T))
        self.uncMean = self.Theta

    def calcFC(self):
        self.Ft = np.exp(-self.K * self.dt)
        self.Ct = np.dot(self.Theta, np.eye(4)) - np.dot(self.Theta, np.exp(-self.K * self.dt))

    def kalmanfilter(self):
        para = self.para
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

        self.checkEigen()
        self.calcAB()
        self.condUncCov()
        self.calcFC()

        for o in range(0, self.obs):
            # prediction step
            self.Xt_1 = np.dot(self.Ft, self.uncMean) + self.Ct
            self.Pt_1 = np.dot(self.Ft, np.dot(self.unconVar, self.Ft.T)) + np.dot(np.exp(-self.K * self.dt),
                                                                                   np.dot(self.Sigma,
                                                                                          np.dot(self.Sigma.T,
                                                                                                 np.exp(
                                                                                                     -self.K *
                                                                                                     self.dt).T))
                                                                                   )
            # model implied yields
            self.yt = -self.A + (np.dot(-self.Bmatrix, self.Xt_1))

            # residuals
            self.res = np.array(self.observedyield.iloc[o]) - self.yt

            # cov matrix prefit residuals
            self.S = self.H + np.dot(self.Bmatrix, np.dot(self.Pt_1, self.Bmatrix.T))

            # det of cov matrix prefit residuals
            self.detS = np.linalg.det(self.S)
            if self.detS > 0:
                # inverse of cov matrix
                self.Sinv = np.linalg.inv(self.S)

                # kalman gain matrix
                self.gainMatrix = np.dot(self.Pt_1, np.dot(self.Bmatrix.T, self.Sinv))

                # updated step
                self.Xt = self.Xt_1 + np.dot(self.gainMatrix, self.res.T)
                self.Pt = np.dot(np.eye(4), self.Pt_1) - np.dot(self.gainMatrix, np.dot(self.Bmatrix, self.Pt_1))

                # log likelihood for each obs step
                self.ith_loglikelihood = self.loglike - 0.5 * (
                        np.log(self.detS) + np.dot(self.res, np.dot(self.Sinv, self.res.T)))

                # update the loglikelihood
                self.loglike += self.ith_loglikelihood

            else:
                print('Determinant is not-positive.')
                break

        return -self.loglike
