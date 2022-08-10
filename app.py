
from random import randint

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from dash import Dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output

# ----------------------------------------------------------------------------------------------------------------------

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Apparent Loss in Water")
            ],
            style = {"text-align": "center"}
        ),
        html.Br(),
        
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        html.H5("File Location"),
                        dcc.Input(id="input-file-location", 
                                  placeholder="directory/file.csv", 
                                  type="text", value="water_cons_data.csv")
                    ]
                ),
                dbc.Col(
                    children=[
                        html.H5("Time Range"),
                        dcc.RadioItems(id="input-radio-item",
                                       options=[dict(label=' 6 Months', value=6), 
                                                dict(label=' 1 Year', value=12),
                                                dict(label=' 2 Years', value=24)], 
                                       value=12, labelStyle={'display': 'block'})
                    ]
                ),
                dbc.Col(
                    children=[
                        html.H5("Sample Size"),
                        dcc.Input(id="input-sample-size", placeholder='Enter a %', 
                                  type='number', value=10)
                    ]
                )
            ]
        ),
        html.Br(),
        
        html.Div(
            [
              html.H3("K-means Clustering")  
            ],
            style = {"text-align": "center"}
        ),
        html.Br(),
        
        dbc.Row(
            children=[
                dbc.Col(
                    children=[
                        html.H5("K-Clusters Range"),
                        dcc.Slider(min=2, max=12, step=1, value=2, id="k-slider"),
                        dcc.Graph(id="pca-scatter")    
                    ],
                    width=6
                ),
                dbc.Col(
                    children=[
                        html.H5("Model Metrics"),
                        dcc.Dropdown(id='metrics-dropdown', 
                                     options= [dict(label='Inertia', value=True),
                                               dict(label='Silhouette Scores', value=False)], 
                                     value=True),
                        dcc.Graph(id="scores-line-plot")
                    ],
                    width=6
                )
            ]
        ),
        html.Br(),
        
        html.Div(
            [
                html.H3("Clusters Characteristics")
            ],
            style = {"text-align": "center"}
        ),
        html.Br(),
        
        dbc.Row(
            children=[
                dbc.Col(dcc.Graph(id="bar-plot"), width=6),
                dbc.Col(dcc.Graph(id="groups-time-plot"), width=6)
            ]
        )
    ]
)


# ----------------------------------------------------------------------------------------------------------------------

def get_frame(fpath="water_cons_data.csv", n_size=10, t_range=12):
    df = pd.read_csv(fpath)
    df.set_index("customer_identifier", inplace=True)
    df_sample = df.sample(frac=n_size * 0.01, random_state=42)
    df = df_sample.iloc[:, -t_range:-1].copy()
    df["2021-Jun"] = df_sample["2021-Jun"].copy()
    
    return df

# ----------------------------------------------------------------------------------------------------------------------

def get_model(fpath="water_cons_data.csv", n_size=10, t_range=12, k=2):
    model = make_pipeline(
        StandardScaler(), KMeans(n_clusters=k, random_state=42)
    )
    model.fit(
        get_frame(fpath=fpath, n_size=n_size, t_range=t_range)
    )
    
    return model

# ----------------------------------------------------------------------------------------------------------------------

def get_pca_labels(fpath="water_cons_data.csv", n_size=10, t_range=12, k=2):
    transformer = PCA(n_components=2, random_state=42)
    X_t = transformer.fit_transform(
        get_frame(fpath=fpath, n_size=n_size, t_range=t_range)
    )
    X_pca = pd.DataFrame(X_t, columns=["PCA1", "PCA2"])
    model = get_model(fpath=fpath, n_size=n_size, t_range=t_range, k=k)
    X_pca["label"] = model.named_steps["kmeans"].labels_.astype(str)
    
    X_pca.sort_values("label", inplace=True)
    
    return X_pca


# ----------------------------------------------------------------------------------------------------------------------

@app.callback(
    Output("pca-scatter", "figure"),
    Input("input-file-location", "value"),
    Input("input-radio-item", "value"),
    Input("input-sample-size", "value"),
    Input("k-slider", "value")
)

def serve_scatter(fpath="water_cons_data.csv", n_size=10, t_range=12, k=2):
    fig = px.scatter(
        data_frame=get_pca_labels(fpath=fpath, n_size=n_size, t_range=t_range, k=k),
        x="PCA1", y="PCA2", color="label",
        title="PCA Representation of Clusters"
    )
    fig.update_layout(xaxis_title="PCA1", yaxis_title="PCA2")
    
    return fig


# ----------------------------------------------------------------------------------------------------------------------


@app.callback(
    Output("scores-line-plot", "figure"),
    Input("input-file-location", "value"),
    Input("input-radio-item", "value"),
    Input("input-sample-size", "value"),
    Input("metrics-dropdown", "value")
)
def serve_metrics_plot(fpath="water_cons_data.csv", n_size=10, t_range=12, iner_met="True"):
    n_clusters = range(2, 13)
    inertia_errors = []
    silhouette_scores = []

    for k in n_clusters:
        model = get_model(fpath=fpath, n_size=n_size, t_range=t_range, k=k)
        inertia_errors.append(model.named_steps["kmeans"].inertia_)
        silhouette_scores.append(
            silhouette_score(
                get_frame(fpath=fpath, n_size=n_size, t_range=t_range), 
                model.named_steps["kmeans"].labels_)
        )
    
    if iner_met:
        metric = "Inertia"
        errors = inertia_errors
    else:
        metric = "Silhouette Score"
        errors = silhouette_scores
    
    fig = px.line(
            x=n_clusters, y=errors, 
            title=f"K-Means Model: {metric} vs Number of Clusters"
          )

    fig.update_layout(xaxis_title="Number of Clusters (k)", yaxis_title=f"{metric}")
    
    return fig


# ----------------------------------------------------------------------------------------------------------------------

@app.callback(
    Output("bar-plot", "figure"),
    Input("input-file-location", "value"),
    Input("input-radio-item", "value"),
    Input("input-sample-size", "value"),
    Input("k-slider", "value")
)
def serve_bar_plot(fpath="water_cons_data.csv", n_size=10, t_range=12, k=2):
    df = get_frame(fpath=fpath, n_size=n_size, t_range=t_range)
    avg_df = df.mean(axis=1).to_frame()
    avg_df.rename(columns={0: 'avg'}, inplace=True)
    model = get_model(fpath=fpath, n_size=n_size, t_range=t_range, k=k)
    avg_df["label"] = model.named_steps["kmeans"].labels_.astype(str) 
    gr_df = avg_df.groupby(["label"]).mean().sort_values(by=['avg'], ascending=True)
    
    fig = px.bar(
        data_frame=gr_df, x=gr_df['avg'], y=gr_df.index,
        orientation='h',
        title="Average Consumptions Across Clusters"
    )
    
    fig.update_layout(xaxis_title="Average", yaxis_title="Cluster")
    
    return fig

# ----------------------------------------------------------------------------------------------------------------------


@app.callback(
    Output("bar-plot", "figure"),
    Input("input-file-location", "value"),
    Input("input-radio-item", "value"),
    Input("input-sample-size", "value"),
    Input("k-slider", "value")
)
def serve_bar_plot(fpath="water_cons_data.csv", n_size=10, t_range=12, k=2):
    df = get_frame(fpath=fpath, n_size=n_size, t_range=t_range)
    avg_df = df.mean(axis=1).to_frame()
    avg_df.rename(columns={'0': 'avg'}, inplace=True)
    model = get_model(fpath=fpath, n_size=n_size, t_range=t_range, k=k)
    avg_df["label"] = model.named_steps["kmeans"].labels_.astype(str) 
    gr_df = avg_df.groupby(["label"]).mean().sort_values(by=['mean'], ascending=False)
    
    fig = px.bar(
        x=gr_df.index, y=gr_df['mean'],
        title="Average Consumption Across Clusters"
    )
    
    fig.update_layout(xaxis_title="Cluster", yaxis_title="Average")
    
    return fig


# ----------------------------------------------------------------------------------------------------------------------

if __name__=="__main__":
    app.run_server(debug=False)

# ----------------------------------------------------------------------------------------------------------------------










