
from random import randint

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from dash import Dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

# ----------------------------------------------------------------------------------------------------------------------

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Apparent Loss in Water")
            ],
            style = {"text-align": "center"}
        ),

        html.H2("File Location"),
        dcc.Input(id="input-file-location", placeholder="/filepath/file.csv",
                  type="text", value="water_cons_data.csv"),

        html.H2("Time Range"),
        dcc.RadioItems(id="input-radio-item",
                       options=[dict(label='6 Months', value=6),
                                dict(label='1 Year', value=12),
                                dict(label='2 Years', value=24)],
                       value=12, labelStyle={'display': 'inline-block'}),

        html.H2("Sample Size"),
        dcc.Input(id="input-sample-size", placeholder='Enter a %', type='number', value=10),

        html.H2("K-means Clustering"),
        dcc.Slider(min=2, max=12, step=1, value=2, id="k-slider"),
        html.Div(id="k-cluster-output-text"),

        dcc.Graph(id="pca-scatter"),

        dcc.Graph(id="groups-line-plot")
    ]
)


# ----------------------------------------------------------------------------------------------------------------------

def get_frame(filepath="water_cons_data.csv", sample_size=10, time_range=24):
    df = pd.read_csv(filepath)
    df.set_index("customer_identifier", inplace=True)
    df_sample = df.sample(frac=sample_size * 0.01, random_state=42)
    df = df_sample.iloc[:, -time_range:-1]
    df["2021-Jun"] = df_sample["2021-Jun"].copy()

    return df

# ----------------------------------------------------------------------------------------------------------------------

def get_model(filepath="water_cons_data.csv", sample_size=10, time_range=24, k=2):
    model = make_pipeline(
        StandardScaler(), KMeans(n_clusters=k, random_state=42)
    )
    model.fit(
        get_frame(filepath=filepath, sample_size=sample_size, time_range=time_range)
    )

    return model

# ----------------------------------------------------------------------------------------------------------------------

def get_pca_labels(filepath="water_cons_data.csv", sample_size=10, time_range=24, k=2):
    transformer = PCA(n_components=2, random_state=42)
    X_t = transformer.fit_transform(
        get_frame(filepath=filepath, sample_size=sample_size, time_range=time_range)
    )
    X_pca = pd.DataFrame(X_t, columns=["PCA1", "PCA2"])
    model = get_model(filepath=filepath, sample_size=sample_size, time_range=time_range, k=k)
    X_pca["label"] = model.named_steps["kmeans"].labels_.astype(str)

    X_pca.sort_values("label", inplace=True)

    return X_pca

# ----------------------------------------------------------------------------------------------------------------------

@app.callback(
    Output("k-text", "children"),
    Input("k-slider", "value")
)
def serve_k_selected(k=2):
    text = [
        html.H3(f"Number of clusters (K): {k}")
    ]

    return text

# ----------------------------------------------------------------------------------------------------------------------

@app.callback(
    Output("pca-scatter", "figure"),
    Input("input-file-location", "value"),
    Input("input-radio-item", "value"),
    Input("input-sample-size", "value"),
    Input("k-slider", "value")
)
def serve_scatter(filepath="water_cons_data.csv", sample_size=10, time_range=24, k=2):
    fig = px.scatter(
        data_frame=get_pca_labels(filepath=filepath, sample_size=sample_size, time_range=time_range, k=k),
        x="PCA1", y="PCA2", color="label",
        title="PCA Representation of Clusters"
    )
    fig.update_layout(xaxis_title="PCA1", yaxis_title="PCA2")

    return fig


# ----------------------------------------------------------------------------------------------------------------------

@app.callback(
    Output("groups-line-plot", "figure"),
    Input("input-file-location", "value"),
    Input("input-radio-item", "value"),
    Input("input-sample-size", "value"),
    Input("k-slider", "value")
)
def serve_line_plot(filepath="water_cons_data.csv", sample_size=10, time_range=24, k=2):
    df = get_frame(filepath=filepath, sample_size=sample_size, time_range=time_range)
    model = get_model(filepath=filepath, sample_size=sample_size, time_range=time_range, k=k)
    df["label"] = model.named_steps["kmeans"].labels_.astype(str)
    months = np.array(df.columns)

    fig = go.Figure()
    for label in df["label"].unique():
        cons = df[df["label"] == label].iloc[randint(0, 10), :].transpose()

        fig.add_trace(go.Scatter(
            x=months, y=np.array(cons), name=label, mode='lines'))

    fig.update_layout(
        title="Consumption Across Multiple Groups",
        xaxis_title="Month", yaxis_title="Consumption"
    )

    return fig

# ----------------------------------------------------------------------------------------------------------------------

if __name__=="__main__":
    app.run_server(debug=False)

# ----------------------------------------------------------------------------------------------------------------------










app.layout = html.Div([
    dbc.Row(children=[
                dbc.Col(id="card_col",width = 6),
                dbc.Col(id="form", width=6, children=[
                    html.Div([
                        dbc.FormGroup(children=[
                            dbc.Label("Question 1"),
                            dbc.Col(
                                dcc.Input(type="text", id="q1", placeholder="Enter your info"),
                                width=6
                            )
                        ],row=True)
                    ]),
                    html.Br(),
                    html.Div(children=[
                        dbc.FormGroup(children=[
                            dbc.Label("Question 2?"),
                            dbc.Col(
                                dbc.Input(type="text",id="q2",placeholder="Enter your info"),
                                width=6
                            )
                        ],row=True)
                    ]),
                    html.Br(),
                    html.Div([
                        dbc.FormGroup(children=[
                            dbc.Label("Yes/No?"),
                            dbc.Col(
                                dbc.RadioItems(id="q3",options=[{"label": "Yes", "value": 1},
                                                                {"label": "No", "value": 2}
                                                               ]
                                              ),width=6
                            )
                        ],row=True)

                    ]),
                    html.Br(),
                    html.Div([
                        html.Button(id='submit-button',
                            n_clicks=0,
                            children='Submit Query',
                            style={'fontSize':24})

                    ])

                ]) #End of second column
                ]), #End of row,
    dbc.Row(
        html.Div([
            dcc.Graph(id='graph1')
        ])
    
    )
    

])


@app.callback(
    Output('card_col','children'),
    Input('submit-button','n_clicks'),
    State('q1','value'),
    State('q2','value'),
    State('q3','value'))
def update_cards(n_clicks,input1,input2,input3):
    
    card1 = dbc.Card([
        dbc.CardBody([
            html.H4(f"{input1}", className="card-title"),
            html.P(f"{input1} was submitted.")
        ],style={'display': 'inline-block',
           'width': '33.3%',
           'text-align': 'center',
           'background-color': 'rgba(37, 150, 190)',
           'color':'white',
           'border': "2px solid white"})
    ])
        
    card2 = dbc.Card([
        dbc.CardBody([
            html.H4(f"{input2}", className="card-title"),
            html.P(f"{input2} was submitted.")
        ],style={'display': 'inline-block',
           'width': '33.3%',
           'text-align': 'center',
           'background-color': 'rgba(37, 150, 190)',
           'color':'white',
           'border': "2px solid white"})
    ])
    
    card3 = dbc.Card([
        dbc.CardBody([
            html.H4(f"{input3}", className="card-title"),
            html.P(f"{input3} was submitted.")
        ],style={'display': 'inline-block',
           'width': '33.3%',
           'text-align': 'center',
           'background-color': 'rgba(37, 150, 190)',
           'color':'white',
           'border': "2px solid white"})
    ])    
        
    return (card1, card2, card3)
    

if __name__ == "__main__":
    app.run_server()