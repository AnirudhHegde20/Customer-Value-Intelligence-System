# src/rfm_features.py

from datetime import datetime
import pandas as pd


def compute_rfm(df: pd.DataFrame, reference_date: datetime | None = None) -> pd.DataFrame:
    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    return rfm


def _map_category(desc: str) -> str:
    desc = str(desc).lower()
    if any(k in desc for k in ["bag", "wallet", "purse"]): return "Bags"
    if any(k in desc for k in ["mug", "cup", "plate", "bowl"]): return "Kitchen"
    if any(k in desc for k in ["lamp", "candle", "lantern"]): return "Home Decor"
    if any(k in desc for k in ["toy", "party", "game"]): return "Toys"
    return "Other"


def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Category"] = df["Description"].apply(_map_category)

    cat = (
        df.groupby(["CustomerID", "Category"])["TotalPrice"]
        .sum()
        .groupby(level=0)
        .apply(lambda x: x / x.sum())
        .reset_index()
        .pivot(index="CustomerID", columns="Category", values="TotalPrice")
        .fillna(0)
    )

    cat.columns = [f"CatShare_{c}" for c in cat.columns]

    return cat.reset_index()


def add_demographics(df: pd.DataFrame) -> pd.DataFrame:
    cc = (
        df.groupby(["CustomerID", "Country"])["TotalPrice"]
        .sum()
        .reset_index()
    )

    idx = cc.groupby("CustomerID")["TotalPrice"].idxmax()
    primary = cc.loc[idx, ["CustomerID", "Country"]]
    primary.rename(columns={"Country": "PrimaryCountry"}, inplace=True)

    primary["IsUK"] = (primary["PrimaryCountry"] == "United Kingdom").astype(int)

    return primary


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rfm = compute_rfm(df)
    cat = add_category_features(df)
    demo = add_demographics(df)

    return (
        rfm
        .merge(cat, on="CustomerID", how="left")
        .merge(demo, on="CustomerID", how="left")
    )
