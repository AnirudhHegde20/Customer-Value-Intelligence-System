# src/data_prep.py

from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/ecommerce_data.csv")  # keep the same path/name


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw ecommerce transactions from an Excel-formatted file
    (dataset is actually XLSX content even though extension is .csv).

    Expected columns:
    InvoiceNo, StockCode, Description, Quantity,
    InvoiceDate, UnitPrice, CustomerID, Country
    """
    df = pd.read_excel(path)  # <-- IMPORTANT: read_excel, not read_csv
    return df



def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean transaction data:
    - drop rows without CustomerID
    - drop negative quantities/prices
    - parse dates
    - normalize Country values
    - compute TotalPrice
    - remove duplicates
    """
    df = df.copy()

    # keep only rows with customer id
    df = df.dropna(subset=["CustomerID"])

    # remove returns / bad values
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # types
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(str)

    # --- Country cleaning ---
    # strip whitespace, unify case
    df["Country"] = df["Country"].astype(str).str.strip()

    # values that mean "no real country"
    bad_countries = ["", "nan", "NaN", "NONE", "None", "Unspecified", "UNSPECIFIED"]

    df["Country"] = df["Country"].replace(bad_countries, pd.NA)

    # if you want to completely drop rows with no country, uncomment:
    # df = df.dropna(subset=["Country"])

    # remove duplicates
    df = df.drop_duplicates()

    # revenue per line
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df
