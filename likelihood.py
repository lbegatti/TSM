from scipy import optimize


#def MLfunction(funcToOptimize, initguess):
#    return optimize.newton(func=lambda params: funcToOptimize, x0=initguess)
def ML(funcToOptimize, initguess):
    f = optimize.minimize(fun=lambda pars: funcToOptimize, x0=initguess, method='nelder-mead')
    return f
