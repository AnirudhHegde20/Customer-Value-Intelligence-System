import pandas as pd
import streamlit as st
import altair as alt

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/rfm_segments.csv")

def main():
    st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
    st.title("Customer Segmentation using RFM + CLV")

    df = load_data()

    cluster_cols = [c for c in df.columns if c.startswith("Cluster_")]
    cluster = st.selectbox("Choose cluster method", cluster_cols)

    st.subheader("Segment Summary")

    metrics = df.groupby(cluster).agg(
        Customers=("CustomerID", "nunique"),
        Revenue=("Monetary", "sum"),
        AvgRecency=("Recency", "mean"),
        AvgFrequency=("Frequency", "mean"),
        AvgCLV=("CLV_6m", "mean"),
    ).reset_index()

    st.dataframe(metrics)

    st.subheader("Recency vs Frequency")
    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x="Recency",
            y="Frequency",
            color=cluster,
            tooltip=["CustomerID", "Monetary", "CLV_6m"]
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

if __name__ == "__main__":
    main()
