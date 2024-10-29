import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc
from projection.projection_generator import ProjectionGenerator
from sampling.sample_generator import SampleGenerator

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], title='High Dimensional Data Visualization')

app.layout = html.Div([
    # navbar
    dbc.NavbarSimple(
        brand='High Dimensional Data Visualization',
        brand_href='#',
        color='#119DFF',
        dark=True,
    ),
    # main container
    dbc.Container([
        dbc.Row([
            dbc.Col([
                # visualization tabs
                dbc.Tabs([
                    # 1D tab
                    dbc.Tab([
                        dbc.Container([
                            # first row 
                            dbc.Row([
                                dbc.Col(
                                    # PCA 1D 
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-pca-1d', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                                dbc.Col(
                                    # t-SNE 1D 
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-tsne-1d', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                            ], className='mb-4'),
                            # second row 
                            dbc.Row([
                                dbc.Col(
                                    # UMAP 1D 
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-umap-1d', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                                dbc.Col(
                                    # axis-aligned 1D 
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-axis-aligned-1d', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                            ])
                        ], class_name='tab-container')
                    ], label='1D'),
                    # 2D tab
                    dbc.Tab([
                        dbc.Container([
                            # first row 
                            dbc.Row([
                                dbc.Col(
                                    # PCA 2D 
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-pca-2d', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                                dbc.Col(
                                    # t-SNE 2D 
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-tsne-2d', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                            ], className='mb-4'),
                            # second row 
                            dbc.Row([
                                dbc.Col(
                                    # UMAP 2D 
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-umap-2d', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                                dbc.Col(
                                    # axis-aligned 2D 
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-axis-aligned-2d', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                            ])
                        ], class_name='tab-container')
                    ], label='2D'),
                    # SPLOM tab
                    dbc.Tab([
                        dbc.Container([
                            # first row 
                            dbc.Row([
                                dbc.Col(
                                    # PCA SPLOM
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-pca-splom', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                                dbc.Col(
                                    # t-SNE SPLOM
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-tsne-splom', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                            ], className='mb-4'),
                            # second row 
                            dbc.Row([
                                dbc.Col(
                                    # UMAP SPLOM
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-umap-splom', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                                dbc.Col(
                                    # axis-aligned SPLOM
                                    dcc.Loading(
                                        html.Div(
                                            id='placeholder-axis-aligned-splom', 
                                            className='placeholder-box',
                                        )
                                    ), 
                                    width=6
                                ),
                            ])
                        ], class_name='tab-container')
                    ], label='SPLOM')
                ])
            ], width=9),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        # sampling method label
                        dbc.Label(
                            'Sampling method:',
                            class_name='sampling-label' 
                        ),
                        # sampling method dropdown
                        dcc.Dropdown(
                            id='sampling-method-dropdown',
                            options=[
                                {'label': 'Lattice Sampling', 'value': 'lattice'},
                                {'label': 'Latin Hypercube', 'value': 'hypercube'},
                                {'label': 'random (uniform)', 'value': 'uniform'},
                                {'label': 'random (Gaussian)', 'value': 'gaussian'},
                                {'label': 'pseudo-random (Fibonacci)', 'value': 'fibonacci'},
                                {'label': 'pseudo-random (Rank - 1 - Lattice)', 'value': 'rank1'}
                            ],
                            value=None,
                            placeholder='Select',
                            className='dropdown-input'
                        )
                    ], width=8, className='offset-md-2 mb-2'),
                ], className='mt-5'),
                dbc.Row([
                    dbc.Col([
                        # dimensionality label
                        dbc.Label(
                            'Dimensionality (3D - 10D):',
                            class_name='dimensionality-label'          
                        ),
                        # dimensionality dropdown
                        dcc.Dropdown(
                            id='dimensionality-dropdown',
                            options=[
                                {'label': '3D', 'value': 3},
                                {'label': '4D', 'value': 4},
                                {'label': '5D', 'value': 5},
                                {'label': '6D', 'value': 6},
                                {'label': '7D', 'value': 7},
                                {'label': '8D', 'value': 8},
                                {'label': '9D', 'value': 9},
                                {'label': '10D', 'value': 10}
                            ],
                            value=None,
                            placeholder='Select',
                            className='dropdown-input'
                        )
                    ], width=8, className='offset-md-2 mb-2'),
                ]),
                dbc.Row([
                    dbc.Col([
                        # number of points label
                        dbc.Label(
                            'Number of Points:',
                            class_name='number-label'
                        ),
                        # number of points dropdown
                        dcc.Dropdown(
                            id='number-of-points-input',
                            options=[
                                {'label': '100', 'value': 100},
                                {'label': '500', 'value': 500},
                                {'label': '700', 'value': 700},
                                {'label': '1000', 'value': 1000}
                            ],
                            value=None,
                            placeholder='Select',
                            className='dropdown-input'
                        )
                    ], width=8, className='offset-md-2 mb-2'),
                ]),
                dbc.Row([
                    dbc.Col([
                        # generate visualization button
                        html.Button(
                            'Generate Visualization', 
                            id='generate-button', 
                            n_clicks=0,
                            className='generate-visualization-button'
                        ),
                        # warning label
                        html.Label(
                            'Please select a sampling method, the dimensionality and the number of points.', 
                            id='warning-label',
                            className='warning-label'
                        ),
                        # discrepancy label
                        html.Label(
                            '', 
                            id='discrepancy-label',
                            className='discrepancy-label'
                        ),
                        # closest neighbor label
                        html.Label(
                            '', 
                            id='closest-neighbor-label',
                            className='closest-neighbor-label'
                        ),
                        # max min distance label
                        html.Label(
                            '', 
                            id='max-min-distance-label',
                            className='max-min-distance-label'
                        ),
                    ], width=8, className='offset-md-2 mt-2'),
                ])
            ], width=3)
        ], justify='center')
    ], fluid=True, className='py-3')
])

@app.callback(
    [Output('placeholder-pca-1d', 'children'),
     Output('placeholder-tsne-1d', 'children'),
     Output('placeholder-umap-1d', 'children'),
     Output('placeholder-axis-aligned-1d', 'children'),
     Output('placeholder-pca-2d', 'children'),
     Output('placeholder-tsne-2d', 'children'),
     Output('placeholder-umap-2d', 'children'),
     Output('placeholder-axis-aligned-2d', 'children'),
     Output('placeholder-pca-splom', 'children'),
     Output('placeholder-tsne-splom', 'children'),
     Output('placeholder-umap-splom', 'children'),
     Output('placeholder-axis-aligned-splom', 'children'),
     Output('warning-label', 'children'),
     Output('discrepancy-label', 'children'),
     Output('closest-neighbor-label', 'children') ,
     Output('max-min-distance-label', 'children')],
    [Input('generate-button', 'n_clicks')],
    [State('sampling-method-dropdown', 'value'),
     State('dimensionality-dropdown', 'value'),
     State('number-of-points-input', 'value')]
)

def update_visualization(n_clicks, sampling_method, dimensionality, number_of_points):
    """
    Updates the visualization placeholders depending on the user input, if some inputs are missing nothing will be generated

    Arguments:
    - n_clicks (type): Number of times the button was clicked
    - sampling_method (str): Name of the selected sampling method
    - dimensionality (int): Number of dimensions the generated data should have
    - number_of_points (int): Number of points the user wants to generate

    Returns:
    - placeholder_list (list(str | Graph)): All visualizations to display on the webpage or an error message for missing inputs
    """
    placeholder_list = [[], [], [], [], [], [], [], [], [], [], [], [], ['Please select a sampling method, the dimensionality and the number of points.'], [], [], []]
    if n_clicks > 0:
        if sampling_method == None:
            placeholder_list[12] = ['Please choose a sampling method!']  
        elif dimensionality == None:
            placeholder_list[12] = ['Please choose a dimensionality!']
        elif number_of_points == None:
            placeholder_list[12] = ['Please choose the number of points!']
        else:    
            visualizations = generate_visualization(sampling_method, dimensionality, number_of_points)
            return visualizations                       
    return placeholder_list

def generate_visualization(sampling_method, dimensionality, number_of_points):
    """
    Generates the visualizations to display on the webpage by first generating the samples 
    and then using different projection methods on them + calculating qualits measuers

    Arguments:
    - sampling_method (str): Name of the selected sampling method
    - dimensionality (int): Number of dimensions the generated data should have
    - number_of_points (int): Number of points the user wants to generate

    Returns:
    - result (list(str | Graph)): All visualizations to display on the webpage or an error message for missing inputs
    """
    sampled_df = SampleGenerator.generate_samples(sampling_method, dimensionality, number_of_points)
    X = sampled_df.iloc[:, :dimensionality].values
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    visualizations = ProjectionGenerator.generate_visualizations(X_standardized)
    quality_measures = SampleGenerator.calculate_quality_measures(X_standardized)
    result = list(visualizations) + [''] + list(quality_measures)
    return result

if __name__ == '__main__':
    app.run_server(debug=True)