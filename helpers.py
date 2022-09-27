import pandas as pd
import plotly.express as px

def plotYield(df:pd.DataFrame,columns: list, type: str, FD: bool):
    if (FD == False):
        fig = px.line(df,x='Date',y=columns, title=type+' Yields').update_layout(xaxis_title='Date',
                                                                             yaxis_title=type+' Yields Level')
    else:
        fig = px.line(df, x='Date', y=columns, title='Stationary '+type + ' Yields').update_layout(xaxis_title='Date',
                                                                                     yaxis_title=type + ' Yields, after FD')
    return fig