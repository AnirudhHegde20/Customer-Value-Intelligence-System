# src/rfm_features.py

from datetime import datetime
import pandas as pd


def compute_rfm(df: pd.DataFrame, reference_date: datetime | None = None) -> pd.DataFrame:
    """
    Compute RFM per customer:
    - Recency: days since last purchase
    - Frequency: number of unique invoices
    - Monetary: total spend
    """
    df = df.copy()

    if reference_date is None:
        reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    return rfm


def _map_category(desc: str) -> str:
    """Simple keyword-based categories from Description."""
    desc = str(desc).lower()
    if any(k in desc for k in ["bag", "wallet", "purse"]):
        return "Bags"
    if any(k in desc for k in ["mug", "cup", "plate", "bowl"]):
        return "Kitchen"
    if any(k in desc for k in ["lamp", "candle", "lantern", "light"]):
        return "HomeDecor"
    if any(k in desc for k in ["toy", "party", "game"]):
        return "Toys"
    return "Other"


def add_category_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each customer, compute share of revenue in each product category.
    Returns a wide table with columns CatShare_* and one row per CustomerID.
    """
    df = df.copy()
    df["Category"] = df["Description"].apply(_map_category)

    # total revenue per CustomerID x Category
    cat = df.groupby(["CustomerID", "Category"])["TotalPrice"].sum()

    # make columns for categories (one row per CustomerID)
    cat = cat.unstack(fill_value=0)  # index: CustomerID, columns: Category

    # convert absolute revenue to share per customer
    row_sums = cat.sum(axis=1)
    cat = cat.div(row_sums, axis=0).fillna(0)

    # rename columns
    cat.columns = [f"CatShare_{c}" for c in cat.columns]

    # bring CustomerID back as a column
    cat = cat.reset_index()  # index -> column "CustomerID"

    return cat


def add_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each customer:
    - PrimaryCountry: where they spend the most
    - IsUK: 1 if primary country is United Kingdom
    """
    df = df.copy()

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
    """
    Combine RFM + category shares + demographics into
    one customer-level feature table.
    """
    rfm = compute_rfm(df)
    cat = add_category_features(df)
    demo = add_demographics(df)

    features = (
        rfm
        .merge(cat, on="CustomerID", how="left")
        .merge(demo, on="CustomerID", how="left")
    )

    return features
