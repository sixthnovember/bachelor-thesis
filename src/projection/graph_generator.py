from dataframe.dataframe_generator import DataFrameGenerator
from dash import dcc
import plotly.express as px
from plotly.subplots import make_subplots

class GraphGenerator:
    
    def generate_1d_graph(X_1d, projection_method):
        """
        Creates a 1D visualization (histogram) of the given data

        Arguments:
        - X_1d (ndarray): Data previously generated by a projection method in 1D
        - projection_method (str): Name of the used projection method

        Returns:
        - dcc.Graph: Graph that can be displayed on the web page
        """    
        df = DataFrameGenerator.generate_df(X_1d, projection_method, '1D')
        fig = px.histogram(
            df, 
            x=df.columns[0]
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=0), 
            height=200, 
            font=dict(size=3), 
            title=projection_method+" - 1D", 
            title_x=0.5,
            title_font=dict(size=12),
            xaxis=dict(
                title_font=dict(size=10),
                tickfont=dict(size=7)
            ),
            yaxis=dict(
                title_font=dict(size=10),
                tickfont=dict(size=7)
            )
        )
        return dcc.Graph(figure=fig)

    def generate_2d_graph(X_2d, projection_method):
        """
        Creates a 2D visualization (scatter plot) of the given data

        Arguments:
        - X_2d (ndarray): Data previously generated by a projection method in 2D
        - projection_method (str): Name of the used projection method

        Returns:
        - dcc.Graph: Graph that can be displayed on the web page
        """
        df = DataFrameGenerator.generate_df(X_2d, projection_method, '2D')
        fig = px.scatter(
            df, 
            x=df.columns[0],
            y=df.columns[1]
        )
        fig.update_traces(marker=dict(size=3))
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=0), 
            height=200, 
            font=dict(size=3), 
            title=projection_method+" - 2D", 
            title_x=0.5, 
            title_font=dict(size=12),
            xaxis=dict(
                title_font=dict(size=10),
                tickfont=dict(size=7)
            ),
            yaxis=dict(
                title_font=dict(size=10),
                tickfont=dict(size=7)
            )
        )
        return dcc.Graph(figure=fig)
    
    def generate_splom_graph(X_splom, projection_method):
        """
        Creates a SPLOM visualization (matrix of scatterplots with histograms as diagoanl elements) of the given data

        Arguments:
        - X_splom (ndarray): Data previously generated by a projection method
        - projection_method (str): Name of the used projection method

        Returns:
        - dcc.Graph: Graph that can be displayed on the web page
        """
        df = DataFrameGenerator.generate_df(X_splom, projection_method, 'SPLOM')
        n = len(df.columns)
        fig = make_subplots(rows=n, cols=n)
        for i in range(n):
            for j in range(n):
                if i == j: 
                    histogram_fig = px.histogram(df, x=df.columns[i], nbins=20)
                    for trace in histogram_fig.data:
                        fig.add_trace(trace, row=i+1, col=j+1)
                else:
                    scatter_fig = px.scatter(df, x=df.columns[j], y=df.columns[i]) 
                    for trace in scatter_fig.data:
                        trace.marker.size = 2
                        fig.add_trace(trace, row=i+1, col=j+1) 
                if i == n - 1:
                    fig.update_xaxes(
                        title_text=df.columns[j], 
                        row=i+1, 
                        col=j+1,                 
                        title_font=dict(size=10),
                        tickfont=dict(size=7)
                    )
                else:
                    fig.update_xaxes(
                        title_text='', 
                        row=i+1, 
                        col=j+1,
                        title_font=dict(size=10),
                        tickfont=dict(size=7)
                    )
                if j == 0:
                    fig.update_yaxes(
                        title_text=df.columns[i], 
                        row=i+1, 
                        col=j+1,
                        title_font=dict(size=10),
                        tickfont=dict(size=7)
                    )
                else:
                    fig.update_yaxes(
                        title_text='', 
                        row=i+1, 
                        col=j+1,
                        title_font=dict(size=10),
                        tickfont=dict(size=7)
                    )
        fig.update_layout(
            margin=dict(l=20, r=20, t=20, b=0), 
            height=200, 
            font=dict(size=3), 
            title=projection_method+" - SPLOM", 
            title_x=0.5, 
            title_font=dict(size=12),
            xaxis=dict(
                title_font=dict(size=10),
                tickfont=dict(size=7)
            ),
            yaxis=dict(
                title_font=dict(size=10),
                tickfont=dict(size=7)
            )
        ) 
        return dcc.Graph(figure=fig)