from typing import Tuple

import pandas as pd
import plotly.express as px
from pandas import DataFrame
from plotly.graph_objs import Figure


def plotYield(df: pd.DataFrame, columns: list, yieldtype: str, FD: bool):
    """ Method to plot both nominal and real yield for specific maturities"""
    if not FD:
        fig = px.line(df, x='Date', y=columns, title=yieldtype + ' Yields').update_layout(
            xaxis_title='Date', yaxis_title=yieldtype + ' Yields Level')
    else:
        fig = px.line(df, x='Date', y=columns, title='Stationary ' + yieldtype + ' Yields').update_layout(
            xaxis_title='Date', yaxis_title=yieldtype + ' Yields, after FD')
    return fig


def BEIrates(nominal_df: pd.DataFrame, real_df: pd.DataFrame) -> pd.DataFrame:

    """ Simple method for calculating and plotting the BEI rates."""

    BEI = pd.DataFrame()
    BEI['Date'] = nominal_df.Date
    BEI['2Y'] = nominal_df.SVENY02 - real_df.TIPSY02
    BEI['5Y'] = nominal_df.SVENY05 - real_df.TIPSY05
    BEI['10Y'] = nominal_df.SVENY10 - real_df.TIPSY10
    # hardcoded columns but I think it is fine since we will use it only once.
    px.line(BEI, x='Date', y=['2Y', '5Y', '10Y'], title='BEI Rates, Nominal-Real Yields').update_layout(
        xaxis_title='Date', yaxis_title='BEI Rates').write_image('Output/BEI Rates, Nominal-Real Yields.png')
    return BEI
