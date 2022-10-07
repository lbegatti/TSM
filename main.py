# Packages
import numpy as np
import pandas as pd
from scipy import optimize

# classes
from datamanipulation import dataManipulation
from helpers import plotYield, BEIrates
from kalmanfilter import KalmanFilter
from nelsonSiegel import NelsonSiegel
from pcaAnalysis import factorAnalysis
from statTests import statTest
from yieldMaturitySelector import yieldMatSelector

# methods - helpers
from likelihood import ML

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
yieldNR = nominal_yields_2_10y_eom.merge(real_yields_2_10y_eom, on='Date').drop('Date', axis=1)
pars = np.random.randn(27)

kf = KalmanFilter(observedyield=yieldNR, obs=len(yieldNR.T), timestep=-1 / 12)

# it breaks if one of the eigen is neg... not sure how to fix it...thou
loglike, Xt, Pt, residuals = kf.kalmanfilter(pars=pars)

## Q12
# ML estimation
MLEstimation = ML(kf.kalmanfilter(pars), np.zeros(27))
final_parameters = MLEstimation.x

## Q13 # here we need to use the parameters after the minimization and re-build the model implied yields with
# the final A and B, because I do not think scipy.minimize can return the residuals directly with new values.
# rmse = np.sqrt((A + BX)-YieldNR

## Q14 using rtN=LNt +StN,rtR=LRt +SRt that come from Xt it should be doable,
# but we need to get the minimization correct first.
