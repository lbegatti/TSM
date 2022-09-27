import pandas as pd

# classes
from datamanipulation import dataManipulation
from helpers import plotYield
from statTests import statTest
from yieldMaturitySelector import yieldMatSelector

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
# plot nominal data
plotYield(nominal_yields_2_10y_eom, columns=['SVENY02', 'SVENY03', 'SVENY05', 'SVENY07', 'SVENY10'],
          yieldtype='Nominal', FD=False).show()
# plot real yield data (TIPS)
plotYield(real_yields_2_10y_eom, columns=['TIPSY02', 'TIPSY03', 'TIPSY05', 'TIPSY07', 'TIPSY10'],
          yieldtype='Real', FD=False).show()

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
          yieldtype='Nominal', FD=True).show()

# check for stationarity in real yields
real2y_yieldADF_FD = stattest.ADFtest(real_yields_2_10y_eom, yieldtype='real', maturity='2y')

# let's try with 5y
real5y_yieldADF_FD = stattest.ADFtest(real_yields_2_10y_eom, yieldtype='real', maturity='5y')

# let's try with 10y
real10y_yieldADF_FD = stattest.ADFtest(real_yields_2_10y_eom, yieldtype='real', maturity='10y')

# real yields are non-stationary, so we re-take the original data, apply first difference and then take only the
# month end
real_yields_2_10y_eom_FD = yieldselector.adjustYieldSerie(real_yields, yieldtype='real', FD=True)

# check stationarity after first difference (only for the 2y)
real2y_yieldADF_FD = stattest.ADFtest(real_yields_2_10y_eom_FD, yieldtype='real', maturity='2y')

# let's plot the nominal yields stationary data
plotYield(real_yields_2_10y_eom_FD, columns=['TIPSY02', 'TIPSY03', 'TIPSY05', 'TIPSY07', 'TIPSY10'], yieldtype='Real',
          FD=True).show()

# Now we have done all the preliminary analysis entailing first differencing the data to ensure stationarity.
# At this point, we can start with the principal components analysis (PCA) using the stationary data.

# PCA
