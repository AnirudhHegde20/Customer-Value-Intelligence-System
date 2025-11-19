# run_pipeline.py

from src.data_prep import load_raw_data, clean_transactions
from src.rfm_features import build_feature_matrix
from src.clustering_models import scale_features, evaluate_kmeans, fit_kmeans, fit_hierarchical, fit_gmm
from src.clv import prepare_summary, fit_clv_models, estimate_clv

import pandas as pd
from pathlib import Path


def main():
    print("Loading raw data...")
    df_raw = load_raw_data()
    df = clean_transactions(df_raw)

    print("Building features...")
    features = build_feature_matrix(df)

    print("Computing CLV...")
    summary = prepare_summary(df)
    bgf, ggf = fit_clv_models(summary)
    clv = estimate_clv(summary, bgf, ggf, months=6).reset_index()

    features = features.merge(clv[["CustomerID", "CLV_6m"]], on="CustomerID", how="left")

    print("Clustering...")
    cluster_cols = ["Recency", "Frequency", "Monetary", "IsUK"]
    cat_cols = [c for c in features.columns if c.startswith("CatShare_")]
    cols = cluster_cols + cat_cols

    X, scaler = scale_features(features, cols)

    eval_df = evaluate_kmeans(X)
    print("KMeans evaluation:\n", eval_df)

    kmeans = fit_kmeans(X, k=4)
    features["Cluster_KMeans"] = kmeans.predict(X)

    _, h_labels = fit_hierarchical(X, k=4)
    features["Cluster_Hierarchical"] = h_labels

    _, gmm_labels = fit_gmm(X, k=4)
    features["Cluster_GMM"] = gmm_labels

    out = Path("data/processed/rfm_segments.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out, index=False)

    print("\nPipeline complete! Saved:", out)


if __name__ == "__main__":
    main()
