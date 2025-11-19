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

    df = df.drop_duplicates()

    # revenue per line
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df
