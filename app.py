
import os

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

def wrangle(filepath):
    df = pd.read_csv(filepath)
    df.set_index("customer_identifier", inplace=True)
    df_sample = df.sample(frac=.1, random_state=42)

    return df_sample

df = wrangle("water_cons_data.csv")


# ----------------------------------------------------------------------------------------------------------------------

app = Dash(__name__)

server = app.server

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1("Apparent Loss in Water")
            ],
            style = {"text-align":"center"}
        ),
        html.H2("K-means Clustering"),
        dcc.Slider(min=2, max=8, step=1, value=2, id="k-slider"),
        html.Div(id="k-text"),
        dcc.Graph(id="pca-scatter")
    ]
)

# ----------------------------------------------------------------------------------------------------------------------

def get_model(k=2):
    model = make_pipeline(
        StandardScaler(), KMeans(n_clusters=k, random_state=42)
    )
    model.fit(df)

    return model

# ----------------------------------------------------------------------------------------------------------------------

def get_pca_labels(k=2):
    transformer = PCA(n_components=2, random_state=42)
    X_t = transformer.fit_transform(df)
    X_pca = pd.DataFrame(X_t, columns=["PCA1", "PCA2"])
    model = get_model(k=k)
    X_pca["labels"] = model.named_steps["kmeans"].labels_.astype(str)

    X_pca.sort_values("labels", inplace=True)

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
    Input("k-slider", "value")
)
def get_scatter(k=2):
    fig = px.scatter(
        data_frame=get_pca_labels(k=k),
        x="PCA1", y="PCA2", color="labels",
        title="PCA Representation of Clusters"
    )
    fig.update_layout(xaxis_title="PCA1", yaxis_title="PCA2")

    return fig

# ----------------------------------------------------------------------------------------------------------------------

if __name__=="__main__":
    app.run_server(debug=False)

# ----------------------------------------------------------------------------------------------------------------------


