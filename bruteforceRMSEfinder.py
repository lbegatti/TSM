from optim import * 

somepars = np.array([0.0030, 0, 0.0, 0.0,
                     0, 0.5730, 0.0, 0.0,
                     0.0, 0.0, 0.2740, 0.0,
                     0.0, 0.0, 0.0, 0.5730,  # KP
                     0.0, 0.0, 0.0, 0.0,  # thetaP
                     0.0154, 0.0117, 0.0154, 0.0117,  # sigma
                     0.2244, 0.28244, 0.0009
                     ])

someresults = kf.kalmanfilter(pars=somepars)

# Finding nice seeds that fulfill pos eigen values and pos det(S):
nice_seeds = {'i': [0], 'initialpars': [somepars], 'initialloglike': [someresults], 'optpars': [],
              'optloglike': [], 'RMSE':[]}

i=13000
while True:
            print(i, end='\r')
            np.random.seed(i)
            initialpars = np.append(np.random.randn(26), 0.01)  # adding jiggle
            loglikefind = kf.kalmanfilter(pars=initialpars)
            if loglikefind < 888888:
                print(f'i: {i}, found loglike: {loglikefind}')
                MLEstimation = optimize.minimize(fun=lambda params: kf.kalmanfilter(pars=params), x0=np.array(initialpars),
                                             method='nelder-mead',
                                             options={'maxiter': 20000, 'maxfev': 10800, 'fatol':1, 'adaptive': True})
                print(f'i:{i}, optimal loglike: {MLEstimation.fun} after {MLEstimation.nit} iterations')
                finalImplYields, finalK, finalTheta, finalSigma, finalXdata = kf.kalmanFilterFinal(MLEstimation.x)      
                RootMeanSquare=RMSE(observedYield=yieldNR[1::], modelYield=finalImplYields[1::], cols=len(yieldNR.columns))
                print(f'RMSE: {RootMeanSquare}')                                
                if all(np.array(RootMeanSquare)<0.0001):
                    final_opt_params = MLEstimation.x
                    with open('Output/final_opt_params.txt', 'w') as filehandle:
                        for param in final_opt_params:
                            filehandle.write(f'{param}\n')
                    with open('Output/initial_opt_params.txt', 'w') as filehandle:
                        for param in initialpars:
                            filehandle.write(f'{param}\n')
                    break
                    
            i += 1




print(nice_seeds)