import pandas as pd
import streamlit as st
import altair as alt


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/processed/rfm_segments.csv")
    return df


def main():
    st.set_page_config(
        page_title="Customer Segmentation using RFM + CLV",
        layout="wide",
    )

    df = load_data()

    # ---------------- Sidebar controls ----------------
    st.sidebar.title("Controls")

    cluster_cols = [c for c in df.columns if c.startswith("Cluster_")]
    if not cluster_cols:
        st.sidebar.error("No cluster columns found. Run run_pipeline.py first.")
        return

    cluster_col = st.sidebar.selectbox("Cluster method", cluster_cols)

    all_countries = sorted(df["PrimaryCountry"].dropna().unique())
    selected_countries = st.sidebar.multiselect(
        "Filter by country",
        options=all_countries,
        default=all_countries,
    )

    min_clv = float(df["CLV_6m"].min()) if "CLV_6m" in df.columns else 0.0
    max_clv = float(df["CLV_6m"].max()) if "CLV_6m" in df.columns else 0.0

    if "CLV_6m" in df.columns:
        clv_filter = st.sidebar.slider(
            "Minimum CLV (6 months)",
            min_value=round(min_clv, 2),
            max_value=round(max_clv, 2),
            value=round(min_clv, 2),
            step=10.0 if max_clv > 1000 else 1.0,
        )
    else:
        clv_filter = None

    min_freq = int(df["Frequency"].min())
    max_freq = int(df["Frequency"].max())
    freq_filter = st.sidebar.slider(
        "Minimum frequency",
        min_value=min_freq,
        max_value=max_freq,
        value=min_freq,
    )

    show_raw = st.sidebar.checkbox("Show raw customer table", value=False)

    # ---------------- Filtered data ----------------
    filtered = df.copy()
    if selected_countries:
        filtered = filtered[filtered["PrimaryCountry"].isin(selected_countries)]
    if clv_filter is not None:
        filtered = filtered[filtered["CLV_6m"] >= clv_filter]
    filtered = filtered[filtered["Frequency"] >= freq_filter]

    # ---------------- Title ----------------
    st.title("Customer Segmentation using RFM + CLV + Clustering")

    st.caption(
        "Explore behavior-based customer segments, their CLV, and geography. "
        "Use the filters on the left to slice the customer base."
    )

    # ---------------- KPI cards ----------------
    kpi1, kpi2, kpi3 = st.columns(3)

    total_customers = filtered["CustomerID"].nunique()
    total_revenue = filtered["Monetary"].sum()
    avg_clv = filtered["CLV_6m"].mean() if "CLV_6m" in filtered.columns else float("nan")

    kpi1.metric("Customers (filtered)", f"{total_customers:,}")
    kpi2.metric("Total Revenue (filtered)", f"{total_revenue:,.0f}")
    if "CLV_6m" in filtered.columns:
        kpi3.metric("Average CLV (6m, filtered)", f"{avg_clv:,.0f}")
    else:
        kpi3.metric("Average CLV (6m, filtered)", "N/A")

    # ---------------- Segment summary ----------------
    st.subheader("Segment summary")

    agg_dict = {
        "CustomerID": "nunique",
        "Monetary": "sum",
        "Recency": "mean",
        "Frequency": "mean",
    }
    if "CLV_6m" in filtered.columns:
        agg_dict["CLV_6m"] = "mean"

    summary = (
        filtered.groupby(cluster_col)
        .agg(agg_dict)
        .reset_index()
        .rename(
            columns={
                "CustomerID": "Customers",
                "Monetary": "TotalRevenue",
                "Recency": "AvgRecency",
                "Frequency": "AvgFrequency",
                "CLV_6m": "AvgCLV",
            }
        )
    )

    st.dataframe(summary, use_container_width=True)

    # ---------------- Auto insights ----------------
    if not summary.empty:
        total_rev_all = summary["TotalRevenue"].sum()
        total_cust_all = summary["Customers"].sum()

        # cluster with highest CLV
        if "AvgCLV" in summary.columns:
            best_clv_row = summary.loc[summary["AvgCLV"].idxmax()]
            best_clv_segment = best_clv_row[cluster_col]
            best_clv_value = best_clv_row["AvgCLV"]
        else:
            best_clv_segment, best_clv_value = None, None

        # cluster with most revenue
        best_rev_row = summary.loc[summary["TotalRevenue"].idxmax()]
        best_rev_segment = best_rev_row[cluster_col]
        best_rev_share = best_rev_row["TotalRevenue"] / total_rev_all if total_rev_all > 0 else 0

        st.markdown("### Key findings (for current filters)")
        bullets = []

        bullets.append(
            f"- **Segment `{best_rev_segment}`** contributes about "
            f"**{best_rev_share:.0%} of filtered revenue**."
        )

        if best_clv_segment is not None:
            bullets.append(
                f"- **Segment `{best_clv_segment}`** has the **highest average CLV** "
                f"(~{best_clv_value:,.0f} over 6 months)."
            )

        bullets.append(
            f"- On average, the filtered view contains **{total_cust_all:,} customers** "
            f"across **{len(summary)} segments**."
        )

        st.markdown("\n".join(bullets))

    # ---------------- Charts ----------------
    st.markdown("---")
    st.subheader("Behavioral patterns")

    col1, col2 = st.columns(2)

    # Recency vs Frequency
    with col1:
        st.markdown("**Recency vs Frequency by segment**")
        chart_rf = (
            alt.Chart(filtered)
            .mark_circle(opacity=0.6)
            .encode(
                x=alt.X("Recency", title="Recency (days since last purchase)"),
                y=alt.Y("Frequency", title="Frequency (number of orders)"),
                color=cluster_col,
                tooltip=[
                    "CustomerID",
                    "Recency",
                    "Frequency",
                    "Monetary",
                    "CLV_6m",
                    "PrimaryCountry",
                ],
            )
            .interactive()
        )
        st.altair_chart(chart_rf, use_container_width=True)

    # Monetary vs CLV
    with col2:
        if "CLV_6m" in filtered.columns:
            st.markdown("**Monetary vs CLV (6m) by segment**")
            chart_mv = (
                alt.Chart(filtered)
                .mark_circle(opacity=0.6)
                .encode(
                    x=alt.X("Monetary", title="Historical Monetary value"),
                    y=alt.Y("CLV_6m", title="Predicted CLV (6 months)"),
                    color=cluster_col,
                    tooltip=[
                        "CustomerID",
                        "Monetary",
                        "CLV_6m",
                        "Recency",
                        "Frequency",
                    ],
                )
                .interactive()
            )
            st.altair_chart(chart_mv, use_container_width=True)
        else:
            st.info("CLV_6m not available in data.")

    # CLV distribution per segment
    if "CLV_6m" in filtered.columns:
        st.subheader("CLV distribution by segment")
        chart_clv_box = (
            alt.Chart(filtered)
            .mark_boxplot()
            .encode(
                x=cluster_col,
                y=alt.Y("CLV_6m", title="CLV (6 months)"),
            )
        )
        st.altair_chart(chart_clv_box, use_container_width=True)

    # Revenue by country
    st.subheader("Revenue by country (filtered)")

    country_rev = (
        filtered.groupby("PrimaryCountry")["Monetary"]
        .sum()
        .reset_index()
        .rename(columns={"Monetary": "Revenue"})
        .sort_values("Revenue", ascending=False)
        .head(15)
    )

    chart_country = (
        alt.Chart(country_rev)
        .mark_bar()
        .encode(
            x=alt.X("Revenue:Q", title="Revenue"),
            y=alt.Y("PrimaryCountry:N", sort="-x", title="Country"),
            tooltip=["PrimaryCountry", "Revenue"],
        )
    )

    st.altair_chart(chart_country, use_container_width=True)

    # ---------------- Raw data (optional) ----------------
    if show_raw:
        st.markdown("---")
        st.subheader("Raw customer-level data (filtered)")
        st.dataframe(filtered, use_container_width=True)


if __name__ == "__main__":
    main()
