from typing import Any

from nelsonSiegel import NelsonSiegel
import numpy as np
from random import gauss


class KalmanFilter(NelsonSiegel):
    """
    Kalman Filter class
        - state transition equation
        - measurement equation
        - conditional / unconditional variance
        - residual variance
        - prediction step moments
        - updated step
    """

    def __init__(self, Lambda: float, sigmaL: float, sigmaS: float, Xt: np.array, tauList: list, k, x0, theta, sigma,
                 start, end, pi: float, T: int, N: int, idMatrix: np.matrix, predEqVal: list or float,
                 predEqVar: list or float, observedYields: list, H):
        # inherit from Nelson Siegel class the parameters to compute the A and B, needed in the measurement eq.
        super().__init__(Lambda, sigmaL, sigmaS, Xt, tauList)

        self.pi = pi  # greek pi so 3.14....
        self.T = T  # total number of observation dates
        self.N = N  # number of obs (yields) per time period t. So we have 1y 5y and 10y per t
        self.idMatrix = idMatrix  # same size as updated step variance 2x2?
        self.predEqExpVal = predEqVal
        self.predEqVar = predEqVar
        self.observedYields = observedYields  # real and nominal 10x1 vector
        self.k = k  # matrix 4x4 with lambda placed on diag (see text in the assignment)
        self.x0 = x0  # initial value -> unconditional mean
        self.theta = theta  # vector 4x1 -> all zeros?
        self.sigma = sigma  # matrix 4x4 for both real and nominal yields
        self.start = start  # start period
        self.end = end  # end period
        self.H = H  # matrix of measurement errors 10x1 vector, i.e. one for each yield...

    def stateTransEq(self, t) -> float:
        """This is the State transition equation (Xt), with initial guess x0"""

        statetranseqt = np.exp(-self.k * t) * self.x0 + np.sum(np.exp(-self.k * (t - 1)) * self.k * self.theta) + \
                        np.sum(np.exp(-self.k * (t - 1)) * self.sigma * gauss(0, 1))

        return statetranseqt

    def condVar(self, t):
        """Conditional variance from Christensen and Lopez paper."""

        conditVar = np.sum(np.exp(-self.k(t - 1)) * self.sigma * self.sigma.T * (np.exp(-self.k * (t - 1))).T)
        # var = np.std([self.statetranseqt(t=t) for t in range(self.start, self.end)][:-1]) ** 2
        return conditVar

    def uncondVar(self):
        """This is the simple variance of state transition equation."""

        var = np.std([self.stateTransEq(t=t) for t in range(self.start, self.end)][:-1]) ** 2
        return var

    def predictionStepMoments(self) -> tuple[Any, ...]:
        """Prediction step equations of the Kalman Filter."""

        self.predEqExpVal = [np.exp(-self.k) * self.stateTransEq(t - 1) + self.x0 * (self.idMatrix - np.exp(-self.k))
                             for t in
                             # np.sum(np.exp(-self.k * (t - 1)) * self.k * self.theta)
                             range(self.start, self.end)]
        self.predEqVar = [np.exp(-self.k) * self.uncondVar() * (np.exp(-self.k)).T +
                          self.condVar(t=t) for t in
                          range(self.start, self.end)]
        return self.predEqExpVal, self.predEqVar

    def measEq(self, t) -> float:
        """Measurement equation"""

        yt = self.A(tau=t) + self.B(tau=t) * self.stateTransEq(t=t) + self.sigma
        return yt

    def updateStepMoments(self) -> tuple[Any, ...]:
        """Updated step equations of the Kalman Filter."""

        updatedEqExpVal = [self.predEqExpVal[t] + self.predEqVar[t] * self.B(t).T *
                           (np.std(self.measEq(t=t)) ** 2) ** (-1) *
                           (self.observedYields[t] - self.measEq(t=t)) for t in range(self.start, self.end)]

        updatedEqVar = [(self.idMatrix - self.predEqVar[t] * self.B(t).T * (np.std(self.measEq(t=t)) ** 2) ** (
            -1)) * self.predEqVar[t] for t in range(self.start, self.end)]

        return updatedEqExpVal, updatedEqVar

    def St(self, t):
        """This is the variance of the pre-fit residuals."""

        prefitResVar = self.H + self.B(tau=t) * self.predEqExpVal() * self.B(tau=t).T

        return prefitResVar

    def residuals(self, t):
        """This returns the pre-fit residual."""

        res = self.observedYields[t] - self.measEq(t=t)

        return res

    def logLikelihood(self) -> float:
        """Log likelihood equation for a Gaussian process."""

        loglikelihood = -(self.N * self.T / 2) * np.log(2 * self.pi) - (1 / 2) * np.sum(
            [np.log(self.St(t=t)) + (self.residuals(t=t)).T * (self.St(t=t) ** (-1)) *
             (self.residuals(t=t)) for t in range(self.start, self.end)])

        return loglikelihood
