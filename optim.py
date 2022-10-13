# Packages
import time
from os.path import exists
from scipy import optimize

# classes
from kalmanfilter import KalmanFilter
from preliminaryAnalysis import *

afnspars = np.array([5.2740, 9.0130, 0.0, 0.0,
                     -0.2848, 0.5730, 0.0, 0.0,
                     0.0, 0.0, 5.2740, 9.0130,
                     0.0, 0.0, -0.2848, 0.5730,  # KP
                     0.0, 0.0, 0.0, 0.0,  # thetaP
                     0.0154, 0.0117, 0.0154, 0.0117,  # sigma
                     0.8244, 0.08244, 0.1
                     ])
somepars = np.array([0.0030, 0, 0.0, 0.0,
                     0, 0.5730, 0.0, 0.0,
                     0.0, 0.0, 0.2740, 0.0,
                     0.0, 0.0, 0.0, 0.5730,  # KP
                     0.0, 0.0, 0.0, 0.0,  # thetaP
                     0.0154, 0.0117, 0.0154, 0.0117,  # sigma
                     0.2244, 0.28244, 0.0009
                     ])

jacobpars = np.array([2.2, 0.3, -0.4, 0.5,
                      -1.0, 0.7, 0.8, 0.4,
                      -0.5, 0.2, 2.2, 1.3,
                      -3.0, 0.5, -1.5, 2.5,  # KP
                      0.02, 0.02, 0.001, 0.001,  # Theta
                      0.001, 0.001, 0.001, 0.001,
                      0.5, 0.5, 0.01])

# Q11 - Kalman filter
print('===============Q11===============')

yieldNR = nominal_yields_2_10y_eom.merge(real_yields_2_10y_eom, on='Date').drop('Date', axis=1)

# Initialized the KF class
kf = KalmanFilter(observedyield=yieldNR, obs=len(yieldNR), timestep=1 / 12)

# KF with some initial values
afnsresults = kf.kalmanfilter(pars=afnspars)
someresults = kf.kalmanfilter(pars=somepars)
jacobresults = kf.kalmanfilter(pars=jacobpars)

# Finding nice seeds that fulfill pos eigen values and pos det(S):
nice_seeds = {'i': [0], 'initialpars': [somepars], 'initialloglike': [someresults], 'optpars': [],
              'optloglike': []}


def optimizationMLE():
    """ Simple method to group the optimization via MLE."""

    if exists('Output/final_opt_params.txt'):
        logger.info("Final parameters already exist, so no need to optimize!")
    else:
        i = 1
        while len(nice_seeds['i']) < 1:
            logger.info('Finding nice seeds...\n')
            np.random.seed(i)
            initialpars = np.append(np.random.randn(-0.1, 0.1, 26), 0.01)  # adding jiggle
            loglikefind = kf.kalmanfilter(pars=initialpars)
            if loglikefind < 888888:
                print(f'i: {i}, found: {len(nice_seeds["i"])}, loglike: {loglikefind}')
                nice_seeds['i'].append(i)
                nice_seeds['initialpars'].append(initialpars)
                nice_seeds['initialloglike'].append(loglikefind)
            i += 1
        print(nice_seeds['i'], nice_seeds['initialloglike'])

        startstamp = time.time()
        for j, pars in enumerate(nice_seeds['initialpars']):
            logger.info(f'Optimizing seed {j} out of {len(nice_seeds[list(nice_seeds.keys())[0]]) - 1}...\n')
            MLEstimation = optimize.minimize(fun=lambda params: kf.kalmanfilter(pars=params), x0=np.array(pars),
                                             method='nelder-mead',
                                             options={'maxiter': 20000, 'maxfev': 10800, 'adaptive': True})
            nice_seeds['optpars'].append(MLEstimation.x)
            logger.info(f'optloglike after {MLEstimation.nit} iterations:{MLEstimation.fun}')
            nice_seeds['optloglike'].append(MLEstimation.fun)
            timestamp = time.time()
            logger.info(time.strftime('%H:%M:%S', time.gmtime(timestamp - startstamp)) + ", " +
                        time.strftime('%H:%M:%S', time.localtime(time.time())))
        endstamp = time.time()
        logger.info("ALL DONE\n")
        logger.info(time.strftime('%H:%M:%S', time.gmtime(endstamp - startstamp)) + ", " +
                    time.strftime('%H:%M:%S', time.localtime(time.time())))

        print(nice_seeds['i'], nice_seeds['initialloglike'], nice_seeds['optloglike'])


def saveOptParams():
    """ Method to store the optimal set of parameters obtained via an iterative MLE process. """

    if exists('Output/final_opt_params.txt'):
        open('Output/final_opt_params.txt', 'r')
        final_opt_params = []
        with open('Output/final_opt_params.txt', 'r') as file:
            for line in file:
                curr_place = line[:-1]
                final_opt_params.append(float(curr_place))

    else:
        final_opt_params = nice_seeds['optpars'][np.argmin(nice_seeds['optloglike'])]
        with open('Output/final_opt_params.txt', 'w') as filehandle:
            for param in final_opt_params:
                filehandle.write(f'{param}\n')

    return final_opt_params
