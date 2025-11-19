# run_pipeline.py

from pathlib import Path

from src.data_prep import load_raw_data, clean_transactions
from src.rfm_features import build_feature_matrix
from src.clustering_models import (
    scale_features,
    evaluate_kmeans,
    fit_kmeans,
    fit_hierarchical,
    fit_gmm,
)
from src.clv import prepare_summary, fit_clv_models, estimate_clv


def main():
    # 1. Load & clean data
    print("Loading raw data...")
    df_raw = load_raw_data()
    df = clean_transactions(df_raw)
    print(f"Cleaned rows: {len(df)}")

    # 2. Build customer features
    print("Building RFM + category + country features...")
    features = build_feature_matrix(df)

    # 3. CLV estimation
    print("Estimating CLV (6 months)...")
    summary = prepare_summary(df[["CustomerID", "InvoiceDate", "TotalPrice"]])
    bgf, ggf = fit_clv_models(summary)
    clv = estimate_clv(summary, bgf, ggf, months=6).reset_index()

    features = features.merge(
        clv[["CustomerID", "CLV_6m"]],
        on="CustomerID",
        how="left"
    )

        # 4. Clustering
    print("Running clustering...")

    # features we'll cluster on
    cluster_cols = ["Recency", "Frequency", "Monetary", "IsUK"]
    cat_cols = [c for c in features.columns if c.startswith("CatShare_")]
    feature_cols = cluster_cols + cat_cols

    # 4a. Handle missing values in clustering features
    # Drop rows where core RFM is missing (should be rare)
    features = features.dropna(subset=["Recency", "Frequency", "Monetary"])

    # For IsUK and category share features, fill NaN with 0
    if "IsUK" in features.columns:
        features["IsUK"] = features["IsUK"].fillna(0)

    for c in cat_cols:
        features[c] = features[c].fillna(0)

    # Optional: quick sanity check (you can comment out later)
    # print("NaNs per feature after cleaning:\n", features[feature_cols].isna().sum())

    # 4b. Scale and run KMeans evaluation
    X, scaler = scale_features(features, feature_cols)

    eval_df = evaluate_kmeans(X, range(2, 9))
    print("KMeans evaluation (k, silhouette, inertia):")
    print(eval_df)


    # pick k = 4 for now
    kmeans = fit_kmeans(X, k=4)
    features["Cluster_KMeans"] = kmeans.predict(X)

    _, h_labels = fit_hierarchical(X, k=4)
    features["Cluster_Hierarchical"] = h_labels

    _, gmm_labels = fit_gmm(X, k=4)
    features["Cluster_GMM"] = gmm_labels

    # 5. Save processed data
    out_path = Path("data/processed/rfm_segments.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_path, index=False)

    print(f"\nDone! Saved processed segments to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
