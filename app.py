
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
        
        html.Div(
            [

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
                        html.H5("K-clusters range"),
                        dcc.Slider(min=2, max=12, step=1, value=2, id="k-slider"),
                        html.Div(id="k-cluster-output-text"),
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
        
        dcc.Graph(id="groups-line-plot")
    ]
)


# ----------------------------------------------------------------------------------------------------------------------

def get_frame(filepath="water_cons_data.csv", sample_size=10, time_range=12):
    df = pd.read_csv(filepath)
    df.set_index("customer_identifier", inplace=True)
    df_sample = df.sample(frac=sample_size * 0.01, random_state=42)
    df = df_sample.iloc[:, -time_range:-1].copy()
    df["2021-Jun"] = df_sample["2021-Jun"].copy()
    
    return df

# ----------------------------------------------------------------------------------------------------------------------

def get_model(filepath="water_cons_data.csv", sample_size=10, time_range=12, k=2):
    model = make_pipeline(
        StandardScaler(), KMeans(n_clusters=k, random_state=42)
    )
    model.fit(
        get_frame(filepath=filepath, sample_size=sample_size, time_range=time_range)
    )
    
    return model

# ----------------------------------------------------------------------------------------------------------------------

def get_pca_labels(filepath="water_cons_data.csv", sample_size=10, time_range=12, k=2):
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
    Output("k-cluster-output-text", "children"),
    Input("k-slider", "value")
)
def serve_k_selected(k=2):
    text = [
        html.H6(f"Number of clusters (K): {k}")
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

def serve_scatter(filepath="water_cons_data.csv", sample_size=10, time_range=12, k=2):
    fig = px.scatter(
        data_frame=get_pca_labels(filepath=filepath, sample_size=sample_size, time_range=time_range, k=k),
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
def serve_metrics_plot(filepath="water_cons_data.csv", sample_size=10, time_range=12, iner_met="True"):
    n_clusters = range(2, 13)
    inertia_errors = []
    silhouette_scores = []

    for k in n_clusters:
        model = get_model(filepath="water_cons_data.csv", sample_size=10, 
                          time_range=12, k=k)
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    silhouette_scores.append(
        silhouette_score(
            get_frame(filepath=filepath, sample_size=sample_size, time_range=time_range), 
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
    Output("groups-line-plot", "figure"),
    Input("input-file-location", "value"),
    Input("input-radio-item", "value"),
    Input("input-sample-size", "value"),
    Input("k-slider", "value")
)
def serve_consumption_plots(filepath="water_cons_data.csv", sample_size=10, time_range=12, k=2):
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










