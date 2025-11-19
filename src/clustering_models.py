# src/clustering_models.py

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np


def scale_features(df: pd.DataFrame, cols: list[str]):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])
    return X, scaler


def evaluate_kmeans(X, k_list=range(2, 9)):
    results = []
    for k in k_list:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        inertia = model.inertia_
        results.append({"k": k, "silhouette": sil, "inertia": inertia})
    return pd.DataFrame(results)


def fit_kmeans(X, k=4):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    return model


def fit_hierarchical(X, k=4):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    return model, labels


def fit_gmm(X, k=4):
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(X)
    return model, labels
