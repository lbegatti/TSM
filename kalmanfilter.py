from typing import Any

from nelsonSiegel import NelsonSiegel
import numpy as np
from random import gauss


class KalmanFilter(NelsonSiegel):
    """
    Kalman Filter class
        - xt
        - pt
        - prediction step moments
        - measurement eq
        - updated step
    """

    def __init__(self, Lambda: float, sigmaL: float, sigmaS: float, Xt: np.array, tauList: list, k, x0, theta, sigma,
                 start, end, pi: float, T: int, N: int, idMatrix: np.matrix, predEqVal: list or float,
                 predEqVar: list or float, realYields: list):
        super().__init__(Lambda, sigmaL, sigmaS, Xt, tauList)
        self.pi = pi
        self.T = T
        self.N = N
        self.idMatrix = idMatrix
        self.predEqExpVal = predEqVal
        self.predEqVar = predEqVar
        self.realYields = realYields
        self.k = k
        self.x0 = x0
        self.theta = theta
        self.sigma = sigma
        self.start = start
        self.end = end

    def xt(self, t) -> float:
        xt = np.exp(-self.k * t) * self.x0 + np.sum(np.exp(-self.k * (t - 1)) * self.k * self.theta) + np.sum(
            np.exp(-self.k * (t - 1)) * self.sigma * gauss(0, 1))

        return xt

    def pt(self):
        var = np.std([self.xt(t=t) for t in range(self.start, self.end)][:-1]) ** 2
        return var

    def predictionStepMoments(self) -> tuple[Any, ...]:
        self.predEqExpVal = [np.exp() * self.xt(t)[:-1] +
                             np.sum(np.exp(-self.k * (t - 1)) * self.k * self.theta) for t in
                             range(self.start, self.end)]
        self.predEqVar = [np.exp(-self.k) * self.pt() * (np.exp(-self.k)).T +
                          np.sum(np.exp(-2 * self.k) * (t - 1)) * self.sigma * self.sigma.T for t in
                          range(self.start, self.end)]
        return self.predEqExpVal, self.predEqVar

    def measEq(self, t) -> float:
        yt = self.A(tau=t) + self.B(tau=t) * self.xt(t=t) + self.sigma
        return yt

    def updateStepMoments(self) -> tuple[Any, ...]:
        updatedEqExpVal = [self.predEqExpVal[t] + self.predEqVar[t] * self.B(t).T *
                           (np.std(self.measEq(t=t)) ** 2) ** (-1) *
                           (self.realYields[t] - self.measEq(t=t)) for t in range(self.start, self.end)]
        updatedEqVar = [(self.idMatrix - self.predEqVar[t] * self.B(t).T * (np.std(self.measEq(t=t)) ** 2) ** (
            -1)) * self.predEqVar[t] for t in range(self.start, self.end)]

        return updatedEqExpVal, updatedEqVar

    def logLikelihood(self) -> list:
        fixpart = -(self.N * self.T / 2) * np.log(2 * self.pi)
        varying = [- (1 / 2) *
                   np.sum(np.log(np.std(self.measEq(t=t)) ** 2) + (self.realYields[t] - self.measEq(t=t)).T *
                          (np.std(self.measEq(t=t)) ** 2) ** (-1)) for t in range(self.start, self.end)]

        logLik = fixpart + varying  # not sure why it is complaining but i will try to fix it ...
        return logLik
