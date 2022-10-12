# Packages
from typing import final
import numpy as np
import pandas as pd
from scipy import optimize
import logging
import time
import plotly.express as px
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
afnspars = np.array([5.2740, 9.0130, 0.0, 0.0,                                                
                      -0.2848, 0.5730, 0.0, 0.0,
                      0.0, 0.0, 5.2740, 9.0130,
                      0.0, 0.0, -0.2848, 0.5730, #KP
                      0.0, 0.0, 0.0, 0.0,  #thetaP
                      0.0154, 0.0117, 0.0154, 0.0117, #sigma
                      0.8244, 0.08244, 0.1
                      ])
kf = KalmanFilter(observedyield=yieldNR, obs=len(yieldNR), timestep=1 / 12)
afnsresults = kf.kalmanfilter(pars=afnspars)

somepars = np.array([0.0030, 0, 0.0, 0.0,                                                
                      0, 0.5730, 0.0, 0.0,
                      0.0, 0.0, 0.2740, 0.0,
                      0.0, 0.0, 0.0, 0.5730, #KP
                      0.0, 0.0, 0.0, 0.0,  #thetaP
                      0.0154, 0.0117, 0.0154, 0.0117, #sigma
                      0.2244, 0.28244, 0.007
                      ])

someresults = kf.kalmanfilter(pars=somepars)

# #Finding nice seeds that fulfill pos eigen values and pos det(S):
nice_seeds = {'i': [0], 'initialpars': [somepars], 'initialloglike': [someresults], 'optpars': [],
              'optloglike': []}

i = 2
print('finding nice_seeds') #i.e. trying to bruteforce, not going so well
while len(nice_seeds['i']) < 1:
    np.random.seed(i)
    initialpars = afnspars + np.append(np.random.uniform(-0.1, 0.1, 26),0)  # adding jiggle
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

if False:#exists('Output/final_opt_params.txt'):
    print("Final paramaters exists will not optimize")
else:
    startstamp = time.time()
    for i, pars in enumerate(nice_seeds['initialpars']):
        logger.info(f'Optimizing seed {i} out of {len(nice_seeds[list(nice_seeds.keys())[0]]) - 1}...')
        MLEstimation = optimize.minimize(fun=lambda params: kf.kalmanfilter(pars=params), x0=pars, method='nelder-mead',
                                         options={'maxiter':2000, 'maxfev':10800, 'adaptive':True})
        # MLEstimation = optimize.differential_evolution(func=lambda params: kf.kalmanfilter(pars=params), bounds=[(-10,10)], x0=pars, maxiter=1000) #, method='nelder-mead'
                                         
        nice_seeds['optpars'].append(MLEstimation.x)
        print(f'optloglike after {MLEstimation.nit} iterations:{MLEstimation.fun}')
        nice_seeds['optloglike'].append(MLEstimation.fun)
        timestamp = time.time()
        logger.info(time.strftime('%H:%M:%S', time.gmtime(timestamp - startstamp)) + ", " +
                    time.strftime('%H:%M:%S', time.localtime(time.time())))
    endstamp = time.time()
    logger.info("ALL DONE\n")
    logger.info(time.strftime('%H:%M:%S', time.gmtime(endstamp - startstamp)) + ", " +
                time.strftime('%H:%M:%S', time.localtime(time.time())))

    print(nice_seeds['i'], nice_seeds['initialloglike'], nice_seeds['optloglike'])

# I GOT THIS XD : [0, 76, 77] [-4554.554683627952, -3869.3091365847804, 878630.5994460909] [-23715964.692126855,
# -23715279.44657981, -19133138.82950106]

print('===============Q12===============')
if False:# exists('Output/final_opt_params.txt'):
    filehandler = open('Output/final_opt_params.txt', 'r')
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

print(final_opt_params)
finalImplYields, finalK, finalTheta, finalSigma, finalXdata = kf.kalmanFilterFinal(final_opt_params)

px.line(finalXdata).write_image('Output/FilteredStateVariables.png')

print('===============Q13===============')
# yieldNR=yieldNR[0:211:]
# finalImplYields=finalImplYields[1:212:]
# here we need to use the parameters after the minimization and re-build the model implied yields with A and B.
rmse = RMSE(observedYield=yieldNR[1::], modelYield=finalImplYields[1::], cols=len(yieldNR.columns))
print(rmse)
px.line(pd.concat([yieldNR[['SVENY02', 'SVENY10']][1::],finalImplYields[[0, 4]][1::]],axis=1)).write_image('Output/NominalObservedVModelYields.png')
px.line(pd.concat([yieldNR[['TIPSY02', 'TIPSY10']][1::],finalImplYields[[5, 9]][1::]],axis=1)).write_image('Output/RealObservedVModelYields.png')

#Plotting observed and fitting yields for 2, 10

print('===============Q14===============')
## Q14 using rtN=LNt +StN,rtR=LRt +SRt that come from Xt it should be doable,


print('===============Q15===============')
rho1 = np.array([1, 1, -1, -1])


def alphamark(beta, k, theta, sigma):
    return (k @ theta).T @ beta + 0.5 * beta.T @ sigma @ sigma.T @ beta

def betamark(k, beta, rho_1):
    # AlphaPrime = (theta @ k).T @ B + 0.5 * B.T @ sigma @ B
    return -rho_1 + k.T @ beta

def RK(funalpha, funbeta, timestep, wa, wb, tau, X):
    # alpha = [wa]
    # beta = [wb]
    alpha = wa
    beta = wb
    obs = int(tau/timestep)
    
    for ob in range(obs):
        k1 = timestep * funbeta(beta)
        k2 = timestep * funbeta(beta + 0.5 * k1)
        k3 = timestep * funbeta(beta + 0.5 * k2)
        k4 = timestep * funbeta(beta + k3)

        l1 = timestep * funalpha(beta)
        l2 = timestep * funalpha(beta + 0.5 * k1)
        l3 = timestep * funalpha(beta + 0.5 * k2)
        l4 = timestep * funalpha(beta + k3)
        
        # beta.append(beta[ob] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
        beta = beta + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # alpha.append(alpha[ob] + (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4))
        alpha = alpha + (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
    return -(alpha+beta.T @ X)/tau

inflation={'2':None, '5':None, '10':None}
tenors=[2,5,10]
for tau in tenors:
    inflation[str(tau)]=[RK(funalpha=lambda B:alphamark(beta=B, k=finalK, theta=finalTheta, sigma=finalSigma), \
                          funbeta=lambda B:betamark(k=finalK, beta=B, rho_1=rho1), \
                          timestep=1/12, \
                          wa=0, \
                          wb=np.zeros(4), \
                          tau=tau, \
                          X=finalXdata.iloc[i].values) \
                          for i in range(len(finalXdata)-1)]


#This is not nice i know quick fix, if you know better please correct
px.line(pd.DataFrame(inflation)).write_image('Output/Inflation.png')

print('===============Q16===============')


irp={'2':None, '5':None, '10':None}
irp['2'] = finalImplYields[0].values-finalImplYields[5].values-np.array(inflation['2'])
irp['5'] = finalImplYields[2].values-finalImplYields[7].values-np.array(inflation['5'])
irp['10']= finalImplYields[4].values-finalImplYields[9].values-np.array(inflation['10'])

px.line(pd.DataFrame(irp)).write_image('Output/InflationRiskPremium.png')


print('===============Q17===============')
#Also making a model BEI:
modelBEI={'2':None, '5':None, '10':None}
modelBEI['2'] = inflation['2']+irp['2']
modelBEI['5'] = inflation['5']+irp['5']
modelBEI['10']= inflation['10']+irp['10']

px.line(pd.DataFrame(modelBEI)).write_image('Output/modelBEI.png')
print('done')



