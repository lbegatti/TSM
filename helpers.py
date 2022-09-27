import pandas as pd
import plotly.express as px


def plotYield(df: pd.DataFrame, columns: list, yieldtype: str, FD: bool):
    if not FD:
        fig = px.line(df, x='Date', y=columns, title=yieldtype + ' Yields').update_layout(
            xaxis_title='Date', yaxis_title=yieldtype + ' Yields Level')
    else:
        fig = px.line(df, x='Date', y=columns, title='Stationary ' + yieldtype + ' Yields').update_layout(
            xaxis_title='Date', yaxis_title=yieldtype + ' Yields, after FD')
    return fig
