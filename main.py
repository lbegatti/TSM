# Packages
from typing import final
import numpy as np
import pandas as pd
from scipy import optimize
import logging
import time
from os.path import exists

# classes
from datamanipulation import dataManipulation
from kalmanfilter import KalmanFilter
from nelsonSiegel import NelsonSiegel
from pcaAnalysis import factorAnalysis
from statTests import statTest
from yieldMaturitySelector import yieldMatSelector

# methods - helpers
from helpers import plotYield, BEIrates, RMSE

# logging
logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.setLevel(logging.INFO)
logger.addHandler(handler)

# Q1 - EDA
print('===============Q1===============')
# instantiate classes
yieldselector = yieldMatSelector()
stattest = statTest()
data_manip = dataManipulation()

# collect data from files
nominal_yields = pd.read_excel('nominal_yields.xlsx').dropna()
real_yields = pd.read_excel('real_yields.xlsx').dropna()

# select specific maturities
nominal_yields_2_10y_eom = yieldselector.adjustYieldSerie(df=nominal_yields, yieldtype='nominal', FD=False)
real_yields_2_10y_eom = yieldselector.adjustYieldSerie(df=real_yields, yieldtype='real', FD=False)

plotgraphs = False
if plotgraphs:
    # plot nominal data
    plotYield(nominal_yields_2_10y_eom, columns=['SVENY02', 'SVENY03', 'SVENY05', 'SVENY07', 'SVENY10'],
              yieldtype='Nominal', FD=False).write_image('Output/Nominal Yields.png')
    # plot real yield data (TIPS)
    plotYield(real_yields_2_10y_eom, columns=['TIPSY02', 'TIPSY03', 'TIPSY05', 'TIPSY07', 'TIPSY10'],
              yieldtype='Real', FD=False).write_image('Output/Real Yields.png')

    # We have extracted and plotted the data, but before proceeding with PCA, it is usually well-practice to
    # ensure stationarity of the data under consideration. From the figures of nominal and real yields respectively,
    # some clear trends which for the statistical inference of data analysis should be removed
    # (or limited as much as possible) for a correct interpretation of the results.

    # check for stationarity in nominal yields
    nominal2y_yieldADF = stattest.ADFtest(nominal_yields_2_10y_eom, yieldtype='nominal', maturity='2y')

    # let's try for 5y yields
    nominal5y_yieldADF = stattest.ADFtest(nominal_yields_2_10y_eom, yieldtype='nominal', maturity='5y')

    # let's try for 10y yields
    nominal10y_yieldADF = stattest.ADFtest(nominal_yields_2_10y_eom, yieldtype='nominal', maturity='10y')

    # nominal yields are non-stationary, so we re-take the original data, apply first difference
    # and then take only the month end.
    nominal_yields_2_10y_eom_FD = yieldselector.adjustYieldSerie(nominal_yields, yieldtype='nominal', FD=True)

    # check stationarity after first difference (only for the 2y)
    nominal2y_yieldADF_FD = stattest.ADFtest(nominal_yields_2_10y_eom_FD, yieldtype='nominal', maturity='2y')

    # let's plot the nominal yields stationary data
    plotYield(nominal_yields_2_10y_eom_FD, columns=['SVENY02', 'SVENY03', 'SVENY05', 'SVENY07', 'SVENY10'],
              yieldtype='Nominal', FD=True).write_image('Output/Stationary Nominal Yield.png')

    # check for stationarity in real yields
    real2y_yieldADF = stattest.ADFtest(real_yields_2_10y_eom, yieldtype='real', maturity='2y')

    # let's try with 5y
    real5y_yieldADF = stattest.ADFtest(real_yields_2_10y_eom, yieldtype='real', maturity='5y')

    # let's try with 10y
    real10y_yieldADF = stattest.ADFtest(real_yields_2_10y_eom, yieldtype='real', maturity='10y')

    # real yields are non-stationary, so we re-take the original data, apply first difference and then take only the
    # month end
    real_yields_2_10y_eom_FD = yieldselector.adjustYieldSerie(real_yields, yieldtype='real', FD=True)

    # check stationarity after first difference (only for the 2y)
    real2y_yieldADF_FD = stattest.ADFtest(real_yields_2_10y_eom_FD, yieldtype='real', maturity='2y')

    # let's plot the nominal yields stationary data
    plotYield(real_yields_2_10y_eom_FD, columns=['TIPSY02', 'TIPSY03', 'TIPSY05', 'TIPSY07', 'TIPSY10'],
              yieldtype='Real',
              FD=True)

    # Now we have done all the preliminary analysis entailing first differencing the data to ensure stationarity.
    # At this point, we can start with the principal components analysis (PCA) using the stationary data.

    # Q2 - PCA
    print('===============Q2===============')
    nominalYieldPCA = factorAnalysis(nominal_yields_2_10y_eom_FD, yieldtype='nominal').pcAnalysis().write_image(
        'Output/Nominal Yield Principal Components.png')
    realYieldPCA = factorAnalysis(real_yields_2_10y_eom_FD, yieldtype='real').pcAnalysis().write_image(
        'Output/Real Yield Principal Components.png')
    # using or not FD dfs does not change the results

    # Q3 - calculate and plot the BEI rates
    print('===============Q3===============')
    BEI = BEIrates(nominal_yields_2_10y_eom, real_yields_2_10y_eom)

    # Q4 - Derive A(tau) and B(tau)
    print('===============Q4===============')
    # Purely theoretical

    # Q5 - Calculate Yield
    print('===============Q5===============')
    nelsonsiegel = NelsonSiegel(Lambda=0.5, sigmaL=0.005, sigmaS=0.01, Xt=np.array([0.02, -0.02]), tauList=[1, 5, 10])
    print(nelsonsiegel.getYields())

    # Q6-Q10
    print('===============Q6-Q10===============')
    # Purely theoretical

# Q11 - Kalman filter
print('===============Q11===============')
yieldNR = nominal_yields_2_10y_eom.merge(real_yields_2_10y_eom, on='Date').drop('Date', axis=1)
# matrixSize = 6
# prova = np.random.rand(matrixSize, matrixSize).reshape(36, 1)
# prova2 = np.dot(prova, prova.transpose())[0:27, 1]
jacobpars = np.array([5.2740, 9.0130, 0.0, 0.0,
                      -0.2848, 0.5730, 0.0, 0.0,
                      0.0, 0.0, 5.2740, 9.0130,
                      0.0, 0.0, -0.2848, 0.5730,  # KP
                      0.0, 0.0, 0.0, 0.0,  # thetaP
                      0.0154, 0.0117, 0.0154, 0.0117,  # sigma
                      0.8244, .08244, 0.1
                      ])
kf = KalmanFilter(observedyield=yieldNR, obs=len(yieldNR), timestep=1 / 12)
jacobresults = kf.kalmanfilter(pars=jacobpars)
# it breaks if one of the eigen is neg... not sure how to fix it...though
# loglike = kf.kalmanfilter(pars=pars)
# print(loglike)

# #Finding nice seeds that fulfill pos eigen values and pos det(S):
nice_seeds = {'i': [0], 'initialpars': [jacobpars], 'initialloglike': [jacobresults], 'optpars': [],
              'optloglike': []}

i = 1
print('finding nice_seeds')
while len(nice_seeds['i']) < 5:
    np.random.seed(i)
    #print(i, end='\r')
    initialpars = jacobpars + np.random.uniform(-0.1, 0.1, 27) #adding jiggle
    loglikefind = kf.kalmanfilter(pars=initialpars)
    if loglikefind < 888888:
        print(f'i: {i}, found: {len(nice_seeds["i"])}, loglike: {loglikefind}')
        nice_seeds['i'].append(i)
        nice_seeds['initialpars'].append(initialpars)
        nice_seeds['initialloglike'].append(loglikefind)
    i += 1
    # if len(nice_seeds['i']) == 3:
    #     break

print(nice_seeds['i'], nice_seeds['initialloglike'])

# Q12
# ML estimation


# def ML(initguess):
#    f = optimize.minimize(fun=lambda pars: kf.kalmanfilter(pars=pars), x0=initguess, method='nelder-mead')
#    return f
if False:#exists('Output/final_opt_params.txt'):
    print("Final paramaters exists will not optimize")
else:
    startstamp = time.time()
    for i, pars in enumerate(nice_seeds['initialpars']):
        logger.info(f'Optimizing seed {i} out of {len(nice_seeds[list(nice_seeds.keys())[0]])-1}...')
        MLEstimation = optimize.minimize(fun=lambda params: kf.kalmanfilter(pars=params), x0=pars, method='nelder-mead') 
        nice_seeds['optpars'].append(MLEstimation.x)
        nice_seeds['optloglike'].append(MLEstimation.fun)
        timestamp = time.time()
        logger.info(time.strftime('%H:%M:%S', time.gmtime(timestamp - startstamp)) + ", " +
                    time.strftime('%H:%M:%S', time.localtime(time.time())))
    endstamp = time.time()
    logger.info("ALL DONE\n")
    logger.info(time.strftime('%H:%M:%S', time.gmtime(endstamp - startstamp)) + ", " +
                time.strftime('%H:%M:%S', time.localtime(time.time())))

    print(nice_seeds['i'], nice_seeds['initialloglike'], nice_seeds['optloglike'])

# I GOT THIS XD : [0, 76, 77] [-4554.554683627952, -3869.3091365847804, 878630.5994460909] [-23715964.692126855, -23715279.44657981, -19133138.82950106]

print('===============Q12===============')
if False:#exists('Output/final_opt_params.txt'):
    filehandler = open('Output/final_opt_params.txt', 'rb')
    final_opt_params=[]
    with open('Output/final_opt_params.txt', 'r') as file:
        for line in file:
            curr_place=line[:-1]
            final_opt_params.append(float(curr_place))

else:
    final_opt_params = nice_seeds['optpars'][np.argmin(nice_seeds['optloglike'])]
    with open('Output/final_opt_params.txt', 'w') as filehandle:
        for param in final_opt_params:
            filehandle.write(f'{param}\n')

print(final_opt_params)
finalXt, finalPt, finalImplYields, finalRes, finalK, finalTheta, finalSigma, finalA, finalBmatrix, \
    finalLambda_N, finalLambda_R = kf.kalmanFilterFinal(final_opt_params)

print('===============Q13===============')

# here we need to use the parameters after the minimization and re-build the model implied yields with A and B.
rmse = RMSE(observedYield=yieldNR, modelYield=finalImplYields, obs=len(yieldNR))
print(rmse)

print('===============Q14===============')
## Q14 using rtN=LNt +StN,rtR=LRt +SRt that come from Xt it should be doable,
# but we need to get the minimization correct first.
