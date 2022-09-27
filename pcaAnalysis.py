import pandas as pd
import plotly.graph_objs as go
from sklearn.decomposition import PCA
import math


class factorAnalysis:
    """Class for factor analysis."""

    def __init__(self, df, yieldtype: str):
        self.df = df
        self.yieldtype = yieldtype

    def pcAnalysis(self):

        """Method to perform PCA and plot the components and variance explained."""

        # nominal yields, after FD
        pca = PCA()
        pca.fit(self.df.iloc[:, 1:])
        total_explained_variation = pca.explained_variance_ratio_.tolist()
        print(f'The explained variance by each PC is: {total_explained_variation}')
        print(f'The sum of the total variation is: {math.fsum(total_explained_variation)}')

        # note multiplying by pca.explained_variance "destroys" the result: level, slope and curvature are not so clear.
        # I remove it, but if necessary we can add it, values are so small anyway.
        YieldComponents = pd.DataFrame(pca.components_.T,
                                       columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'],
                                       index=['2Y', '3Y', '5Y', '7Y', '10Y'])

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=YieldComponents.index, y=YieldComponents.PC1, name="PC1",
                                 # hoverinfo='text+name',
                                 line_shape='spline'))
        fig.add_trace(go.Scatter(x=YieldComponents.index, y=YieldComponents.PC2, name="PC2",
                                 # hoverinfo='text+name',
                                 line_shape='spline'))
        fig.add_trace(go.Scatter(x=YieldComponents.index, y=YieldComponents.PC3, name="PC3",
                                 # hoverinfo='text+name',
                                 line_shape='spline'))

        if self.yieldtype == 'nominal':
            fig.update_layout(title='Nominal Yield Principal components',
                              xaxis_title='Maturities',
                              yaxis_title='Loadings')
        else:
            fig.update_layout(title='Real Yield Principal components',
                              xaxis_title='Maturities',
                              yaxis_title='Loadings')

        fig.show()
        return fig
