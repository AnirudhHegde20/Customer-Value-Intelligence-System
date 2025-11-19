# src/clv.py

import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter


def prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare lifetimes summary.
    df must have: CustomerID, InvoiceDate, TotalPrice
    """
    summary = summary_data_from_transaction_data(
        df,
        customer_id_col="CustomerID",
        datetime_col="InvoiceDate",
        monetary_value_col="TotalPrice",
        observation_period_end=df["InvoiceDate"].max()
    )
    summary = summary[summary["monetary_value"] > 0]
    return summary


def fit_clv_models(summary: pd.DataFrame):
    """
    Fit BG/NBD (frequency) and Gamma-Gamma (monetary value).
    """
    bgf = BetaGeoFitter()
    bgf.fit(summary["frequency"], summary["recency"], summary["T"])

    ggf = GammaGammaFitter()
    ggf.fit(summary["frequency"], summary["monetary_value"])

    return bgf, ggf


def estimate_clv(summary: pd.DataFrame, bgf, ggf, months: int = 6) -> pd.DataFrame:
    """
    Estimate CLV over a given horizon in months.
    """
    summary = summary.copy()
    t = months * 30  # days

    summary["PredPurchases"] = bgf.conditional_expected_number_of_purchases_up_to_time(
        t,
        summary["frequency"],
        summary["recency"],
        summary["T"]
    )

    summary["ExpAvgValue"] = ggf.conditional_expected_average_profit(
        summary["frequency"],
        summary["monetary_value"]
    )

    summary[f"CLV_{months}m"] = summary["PredPurchases"] * summary["ExpAvgValue"]

    return summary
