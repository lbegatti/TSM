import pandas as pd
from statsmodels.tsa.stattools import adfuller

class statTest:

    """Python class for all the statistical tests needed for the assignment."""

    def __init__(self):
        pass

    def ADFtest(self,df: pd.DataFrame,type: str,maturity: str):

        """Augmented Dickey-Fueller test for stationarity."""

        if (type=='real'):
            if (maturity == '2y'):
                df_sub = df.TIPSY02
            elif (maturity == '5y'):
                df_sub = df.TIPSY05
            else:
                df_sub = df.TIPSY10
        else:
            if (maturity == '2y'):
                df_sub = df.SVENY02
            elif (maturity == '5y'):
                df_sub = df.SVENY05
            else:
                df_sub = df.SVENY10

        input = adfuller(df_sub)
        print('ADF Statistic: %f' % input[0])
        print('p-value: %f' % input[1])
        print('Critical Values:')
        for key, value in input[4].items():
            print('\t%s: %.3f' % (key, value))
        if (input[1] > 0.05):
            print("Non-Stationary timeserie")
        else:
            print("Stationary timeserie")

        return input[0],input[1]