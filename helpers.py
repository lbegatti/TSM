import pandas as pd
import plotly.express as px
from statsmodels.tsa.stattools import adfuller

def selectYieldsMaturities(df: pd.DataFrame, type: str,FD: bool) -> pd.DataFrame:
    if (type=='nominal'):
        nominal_yields_2_10y = df[['Date','SVENY02','SVENY03','SVENY05','SVENY07','SVENY10']].apply(lambda x: x/100 if x.name not in 'Date' else x)
        if (FD == False):
            nominal_yields_2_10y_eom = nominal_yields_2_10y[nominal_yields_2_10y['Date'].dt.is_month_end == True].reset_index(inplace=False, drop=True)
            return nominal_yields_2_10y_eom
        else:
            nominal_yields_2_10y_shifted = nominal_yields_2_10y.apply(lambda x: x.diff(1) if x.name not in 'Date' else x).dropna()
            nominal_yields_2_10y_eom_shifted = nominal_yields_2_10y_shifted[
                nominal_yields_2_10y_shifted['Date'].dt.is_month_end == True].reset_index(inplace=False, drop=True)
            return nominal_yields_2_10y_eom_shifted
    else:
        real_yields_2_10y = df[['Date', 'TIPSY02', 'TIPSY03', 'TIPSY05', 'TIPSY07', 'TIPSY10']].apply(
            lambda x: x / 100 if x.name not in 'Date' else x)
        if (FD == False):
            real_yields_2_10y_eom = real_yields_2_10y[real_yields_2_10y['Date'].dt.is_month_end == True].reset_index(inplace=False, drop=True)
            return real_yields_2_10y_eom
        else:
            real_yields_2_10y_shifted = real_yields_2_10y.apply(
                lambda x: x.diff(1) if x.name not in 'Date' else x).dropna()
            real_yields_2_10y_eom_shifted = real_yields_2_10y_shifted[
                real_yields_2_10y_shifted['Date'].dt.is_month_end == True].reset_index(inplace=False, drop=True)
            return real_yields_2_10y_eom_shifted

def plotYield(df:pd.DataFrame,columns: list, type: str, FD: bool):
    if (FD == False):
        fig = px.line(df,x='Date',y=columns, title=type+' Yields').update_layout(xaxis_title='Date',
                                                                             yaxis_title=type+' Yields Level')
    else:
        fig = px.line(df, x='Date', y=columns, title='Stationary '+type + ' Yields').update_layout(xaxis_title='Date',
                                                                                     yaxis_title=type + ' Yields, after FD')
    return fig

def ADFtest(df: pd.DataFrame,type: str, maturity: str):
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