# src/clustering_models.py

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def scale_features(df: pd.DataFrame, cols: list[str]):
    """
    Standardize selected numeric features.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols])
    return X, scaler


def evaluate_kmeans(X: np.ndarray, k_list: Iterable[int] = range(2, 9)) -> pd.DataFrame:
    """
    Evaluate KMeans for different k using silhouette and inertia.
    """
    results = []
    for k in k_list:
        model = KMeans(n_clusters=k, random_state=42)
        labels = model.fit_predict(X)
        sil = silhouette_score(X, labels)
        inertia = model.inertia_
        results.append({"k": k, "silhouette": sil, "inertia": inertia})
    return pd.DataFrame(results)


def fit_kmeans(X: np.ndarray, k: int = 4) -> KMeans:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    return model


def fit_hierarchical(X: np.ndarray, k: int = 4):
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)
    return model, labels


def fit_gmm(X: np.ndarray, k: int = 4):
    model = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(X)
    return model, labels
