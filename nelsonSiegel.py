from array import array
import numpy as np
from pandas._libs.hashtable import Vector


class NelsonSiegel:
    """
    Class for Nelson Siegel Calculated in question 4
    """

    def __init__(self, Lambda: float or np.array or None, sigmaL: float or np.array or None, sigmaS: float or None,
                 Xt: np.array or None, tauList: list) -> None:
        self.Xt = Xt
        self.Lambda = Lambda  # Lambda with cap since lambda is reserved
        self.sigmaL = sigmaL
        self.sigmaS = sigmaS
        self.tauList = tauList

    def A(self, tau):
        bracket = 2 * tau * self.Lambda - 3 + 4 * np.exp(-tau * self.Lambda) - np.exp(
            -2 * tau * self.Lambda)
        return (np.square(self.sigmaL) * np.power(tau, 3)) / 6 + np.square(self.sigmaS) / (
                4 * np.power(self.Lambda, 3)) * bracket

    def Avector(self):
        return np.array([[-self.A(tau=t)/t for t in self.tauList]])

    def B(self, tau):
        return np.array([-tau, (np.exp(-self.Lambda * tau) - 1) / self.Lambda])

    def Bmatrix(self):
        return np.array([-self.B(tau=t)/t for t in self.tauList])

    def getYields(self) -> list:
        return [(t, -self.A(t) / t - np.dot(self.B(t), self.Xt) / t) for t in self.tauList]
