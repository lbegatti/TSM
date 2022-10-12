# Packages

# methods - helpers
from optim import *
from RungeKuttaMethod import *
from preliminaryAnalysis import *

# ML estimation
optimizationMLE()

print('===============Q12===============')
opt_params = saveOptParams()
print(opt_params)

finalImplYields, finalK, finalTheta, finalSigma, finalXdata = kf.kalmanFilterFinal(opt_params)
# plot of filtered state variables after MLE optimization.
filteredStateVariable(finalXdata).write_image('Output/FilteredStateVariables.png')

print('===============Q13===============')
rmse = RMSE(observedYield=yieldNR[1::], modelYield=finalImplYields[1::], cols=len(yieldNR.columns))
print(rmse)

# add the date timestamp for plotting.
finalImplYields['Date'] = nominal_yields_2_10y_eom['Date']
yieldNR['Date'] = nominal_yields_2_10y_eom['Date']
cols = ['implSVENY02', 'implSVENY03', 'implSVENY05', 'implSVENY07', 'implSVENY10', 'implTIPSY02', 'implTIPSY03',
        'implTIPSY05', 'implTIPSY07', 'implTIPSY10', 'Date']
finalImplYields.columns = cols
observedImpliedYields(df_observed=yieldNR, df_implied=finalImplYields).write_image('Output/ObservedVsModelYields.png')

print('===============Q14===============')
## Purely theoretical.

print('===============Q15===============')
inflation = {}
tenors = [2, 5, 10]
for tau in tenors:
    inflation[f'{str(tau)}Y Impl_Infl'] = [
        RK(funalpha=lambda B: alphamark(beta=B, k=finalK, theta=finalTheta, sigma=finalSigma),
           funbeta=lambda B: betamark(k=finalK, beta=B, rho_1=rho1),
           timestep=1 / 12,
           wa=0,
           wb=np.zeros(4),
           tau=tau,
           X=finalXdata.iloc[i].values)
        for i in range(len(finalXdata) - 1)]

infl = pd.DataFrame(inflation)
infl['Date'] = nominal_yields_2_10y_eom['Date']
plotCurve(df=infl, columns=['2Y Impl_Infl', '5Y Impl_Infl', '10Y Impl_Infl'], curvetype='Inflation',
          FD=False).write_image('Output/Inflation.png')

print('===============Q16===============')

irp = IRP(df_implyields=finalImplYields, df_infl=infl)
# Also making a model BEI:
modelBEI = impliedBEI(df_infl=infl, df_irp=irp)

print('===============Q17===============')

logger.info('Assignment completed.')
