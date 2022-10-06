from scipy import optimize
import numpy as np


def MLfunction(funcToOptimize):
    return optimize.minimize(fun=lambda parameters: funcToOptimize, x0=np.array([-100, 100]), method='nelder-mead')
