# Packages
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plotCurve(df: pd.DataFrame, columns: list, curvetype: str, FD: bool):

    """ Method to plot curve: real or nominal yields, expected inflation, for specific maturities."""

    if not FD:
        fig = px.line(df, x='Date', y=columns, title=curvetype + ' Curve').update_layout(
            xaxis_title='Date', yaxis_title=curvetype + ' Curve Level')
    else:
        fig = px.line(df, x='Date', y=columns, title='Stationary ' + curvetype + ' Curve').update_layout(
            xaxis_title='Date', yaxis_title=curvetype + ' Curve, after FD')
    return fig


def BEIrates(nominal_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:

    """ Simple method for calculating and plotting the BEI rates.Columns are hardcoded as only these are relevant."""

    BEI = pd.DataFrame()
    BEI['Date'] = nominal_df.Date
    BEI['2Y'] = nominal_df.SVENY02 - real_df.TIPSY02
    BEI['5Y'] = nominal_df.SVENY05 - real_df.TIPSY05
    BEI['10Y'] = nominal_df.SVENY10 - real_df.TIPSY10
    px.line(BEI, x='Date', y=['2Y', '5Y', '10Y'], title='BEI Rates, Nominal-Real Yields').update_layout(
        xaxis_title='Date', yaxis_title='BEI Rates').write_image('Output/BEI Rates, Nominal-Real Yields.png')

    return BEI


def filteredStateVariable(Xt):

    """ Plot the filtered state variable Xt after the MLE optimization via kalman filter approach."""

    Xt = Xt.rename(columns={0: 'LevelN', 1: 'SlopeN', 2: 'LevelR', 3: 'SlopeR'})

    return px.line(Xt, x=Xt.index, y=Xt.columns,
                   title='Filtered State Variables after MLE').update_layout(
        xaxis_title='obs points', yaxis_title='Filtered State Vars')


def RMSE(observedYield, modelYield, cols):

    """ Method to calculate the Root mean squared error (RMSE) given certain observed and implied parameters."""

    rmse = np.sqrt(np.array([np.mean((modelYield.iloc[:, i] - observedYield.iloc[:, i]) ** 2) for i in range(cols)]))

    return rmse


def observedImpliedYields(df_observed: pd.DataFrame, df_implied: pd.DataFrame):

    """ Plot observed vs implied yields derived by implementation of Kalman Filter and MLE. """

    fig = go.Figure()
    # add observed plots
    fig.add_trace(go.Scatter(x=df_observed['Date'], y=df_observed['SVENY02'], name='SVENY02', mode='lines', opacity=0.5,))
    fig.add_trace(go.Scatter(x=df_observed['Date'], y=df_observed['SVENY10'], name='SVENY10', mode='lines', opacity=0.5,))
    fig.add_trace(go.Scatter(x=df_observed['Date'], y=df_observed['TIPSY02'], name='TIPSY02', mode='lines', opacity=0.5,))
    fig.add_trace(go.Scatter(x=df_observed['Date'], y=df_observed['TIPSY10'], name='TIPSY10', mode='lines', opacity=0.5,))

    # add model implied plots
    fig.add_trace(go.Scatter(x=df_implied['Date'], y=df_implied['implSVENY02'], name='Implied SVENY02', mode='lines'))
    fig.add_trace(go.Scatter(x=df_implied['Date'], y=df_implied['implSVENY10'], name='Implied SVENY10', mode='lines'))
    fig.add_trace(go.Scatter(x=df_implied['Date'], y=df_implied['implTIPSY02'], name='Implied TIPSY02', mode='lines'))
    fig.add_trace(go.Scatter(x=df_implied['Date'], y=df_implied['implTIPSY10'], name='Implied TIPSY10', mode='lines'))

    fig.update_layout(title='2Y and 10Y observed and model implied nominal and real yields',
                      xaxis_title='Date',
                      yaxis_title='Yield')

    return fig


def IRP(df_implyields: pd.DataFrame, df_infl: pd.DataFrame) -> pd.DataFrame:

    """ Inflation Risk Premium: nominal model implied yields - real model implied yields - inflation expectation."""

    inflRiskPrem = pd.DataFrame()
    inflRiskPrem['Date'] = df_implyields['Date']
    inflRiskPrem['2Y'] = df_implyields['implSVENY02'] - df_implyields['implTIPSY02'] - df_infl['2Y_Impl_Infl']
    inflRiskPrem['5Y'] = df_implyields['implSVENY05'] - df_implyields['implTIPSY05'] - df_infl['5Y_Impl_Infl']
    inflRiskPrem['10Y'] = df_implyields['implSVENY10'] - df_implyields['implTIPSY10'] - df_infl['10Y_Impl_Infl']

    plotCurve(df=inflRiskPrem, columns=['2Y', '5Y', '10Y'], curvetype='IRP',
              FD=False).write_image('Output/InflationRiskPremium.png')

    return inflRiskPrem


def impliedBEI(df_infl: pd.DataFrame, df_irp: pd.DataFrame) -> pd.DataFrame:

    """ Model implied BEI rates: inflation expectations + inflation risk premium."""

    modelBEI = pd.DataFrame()
    modelBEI['Date'] = df_infl['Date']
    modelBEI['2Y'] = df_infl['2Y_Impl_Infl'] + df_irp['2Y']
    modelBEI['5Y'] = df_infl['5Y_Impl_Infl'] + df_irp['5Y']
    modelBEI['10Y'] = df_infl['10Y_Impl_Infl'] + df_irp['10Y']
    plotCurve(df=modelBEI, columns=['2Y', '5Y', '10Y'], curvetype='Implied BEI rates', FD=False).write_image(
        'Output/modelBEI.png')

    return modelBEI
