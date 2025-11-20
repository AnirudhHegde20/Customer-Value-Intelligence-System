from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


DATA_PATH = Path("data/processed/rfm_segments.csv")
FIG_DIR = Path("reports/figures")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {DATA_PATH}. "
            f"Run `python run_pipeline.py` first."
        )
    df = pd.read_csv(DATA_PATH)
    return df


def ensure_output_dir():
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def get_default_cluster_col(df: pd.DataFrame) -> str:
    cluster_cols = [c for c in df.columns if c.startswith("Cluster_")]
    if not cluster_cols:
        raise ValueError("No cluster columns found (expected columns starting with 'Cluster_').")
    # prefer KMeans if present
    for c in cluster_cols:
        if "KMeans" in c or "kmeans" in c.lower():
            return c
    return cluster_cols[0]


def plot_rfm_distributions(df: pd.DataFrame):
    """Histogram/KDE-style views for Recency, Frequency, Monetary."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].hist(df["Recency"], bins=40)
    axes[0].set_title("Recency (days)")
    axes[0].set_xlabel("Days since last purchase")

    axes[1].hist(df["Frequency"], bins=40)
    axes[1].set_title("Frequency")
    axes[1].set_xlabel("Number of orders")

    axes[2].hist(df["Monetary"], bins=40)
    axes[2].set_title("Monetary")
    axes[2].set_xlabel("Total spend")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / "rfm_distributions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_segment_revenue_share(df: pd.DataFrame, cluster_col: str):
    """Revenue share by segment."""
    seg_rev = (
        df.groupby(cluster_col)["Monetary"]
        .sum()
        .rename("Revenue")
        .reset_index()
        .sort_values("Revenue", ascending=False)
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(seg_rev[cluster_col].astype(str), seg_rev["Revenue"])
    ax.set_title(f"Revenue Share by Segment ({cluster_col})")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Total revenue")

    ax.bar_label(bars, fmt="%.0f", padding=3, rotation=0)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / "segment_revenue_share.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_clv_by_segment_boxplot(df: pd.DataFrame, cluster_col: str):
    """CLV_6m distribution by segment."""
    if "CLV_6m" not in df.columns:
        print("CLV_6m column not found. Skipping CLV boxplot.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="CLV_6m", by=cluster_col, ax=ax)

    ax.set_title(f"CLV (6m) Distribution by Segment ({cluster_col})")
    ax.set_xlabel("Segment")
    ax.set_ylabel("CLV (6 months)")
    plt.suptitle("")  # remove default pandas title

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = FIG_DIR / "clv_by_segment_boxplot.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_recency_frequency_scatter(df: pd.DataFrame, cluster_col: str):
    """Recency vs Frequency scatter colored by segment."""
    fig, ax = plt.subplots(figsize=(6, 4))
    scatter = ax.scatter(
        df["Recency"],
        df["Frequency"],
        c=df[cluster_col],
        alpha=0.5,
    )
    ax.set_title(f"Recency vs Frequency by Segment ({cluster_col})")
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Frequency (# orders)")
    ax.grid(True, alpha=0.3)

    # Simple legend: unique segments
    handles, labels = scatter.legend_elements()
    ax.legend(handles, labels, title="Segment", loc="best")

    plt.tight_layout()
    out_path = FIG_DIR / "recency_frequency_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_revenue_by_country(df: pd.DataFrame, top_n: int = 10):
    """Top N countries by revenue."""
    if "PrimaryCountry" not in df.columns:
        print("PrimaryCountry not found. Skipping revenue_by_country plot.")
        return

    country_rev = (
        df.groupby("PrimaryCountry")["Monetary"]
        .sum()
        .rename("Revenue")
        .reset_index()
        .sort_values("Revenue", ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(country_rev["PrimaryCountry"], country_rev["Revenue"])
    ax.set_title(f"Top {top_n} Countries by Revenue")
    ax.set_xlabel("Revenue")
    ax.invert_yaxis()  # highest on top

    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = FIG_DIR / "revenue_by_country.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ensure_output_dir()
    df = load_data()
    cluster_col = get_default_cluster_col(df)

    print(f"Using cluster column: {cluster_col}")

    plot_rfm_distributions(df)
    plot_segment_revenue_share(df, cluster_col)
    plot_clv_by_segment_boxplot(df, cluster_col)
    plot_recency_frequency_scatter(df, cluster_col)
    plot_revenue_by_country(df)


if __name__ == "__main__":
    main()
