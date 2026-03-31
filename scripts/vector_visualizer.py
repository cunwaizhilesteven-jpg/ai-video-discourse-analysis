"""
Vector Visualizer - t-SNE / PCA cluster visualization
Outputs PNG + interactive HTML to output/png/ and output/html/
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

SCHEMA_COLORS = {
    "Emotional & Faith Expression":       "#4C72B0",
    "Meme Mockery & Skull Emoji":          "#DD8452",
    "Non-English & Noise":                 "#55A868",
    "Instant Emotional Reaction":          "#C44E52",
    "Celebrity AI Morph Memes":            "#8172B2",
    "Content Criticism & Platform Meta":   "#937860",
    "Family Drama & Plot Twist":           "#DA8BC3",
    "Parenting & Childhood Scenes":        "#8C8C8C",
    "Absurdist & Chaotic Comments":        "#CCB974",
    "Toilet Humor & Food Jokes":           "#64B5CD",
}


def _reduce(matrix: np.ndarray, method: str = "tsne", n_components: int = 2) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=n_components, random_state=42).fit_transform(matrix)
    # t-SNE: PCA pre-reduction for speed on large corpora
    if matrix.shape[1] > 50:
        matrix = PCA(n_components=50, random_state=42).fit_transform(matrix)
    return TSNE(n_components=n_components, random_state=42,
                perplexity=30, n_iter=1000).fit_transform(matrix)


def plot_clusters(matrix: np.ndarray, labels: np.ndarray,
                  method: str = "tsne", sample: int = 5000,
                  out_dir: str = "output") -> None:
    """
    Scatter plot of comment clusters in 2-D reduced space.
    Samples up to `sample` points to keep rendering fast.
    """
    Path(f"{out_dir}/png").mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/html").mkdir(parents=True, exist_ok=True)

    # Sample for speed
    idx = np.random.choice(len(matrix), min(sample, len(matrix)), replace=False)
    mat_s, lab_s = matrix[idx], labels[idx]

    coords = _reduce(mat_s, method)

    unique = sorted(set(lab_s))
    color_map = SCHEMA_COLORS
    colors = [color_map.get(label, "#888888") for label in unique]

    fig, ax = plt.subplots(figsize=(10, 8))
    for label, color in zip(unique, colors):
        mask = lab_s == label
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=label, alpha=0.4, s=8)

    ax.set_title(f"Comment Clusters — {method.upper()} (n={len(mat_s)})")
    ax.legend(title="Schema Dimension", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()

    png_path = f"{out_dir}/png/clusters_{method}.png"
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {png_path}")
    plt.close()

    # Interactive HTML via plotly
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        df_plot = pd.DataFrame({
            "x": coords[:, 0], "y": coords[:, 1], "dimension": lab_s
        })
        fig_px = px.scatter(
            df_plot, x="x", y="y", color="dimension",
            color_discrete_map=SCHEMA_COLORS,
            title=f"Comment Clusters — {method.upper()}",
            labels={"dimension": "Schema Dimension"},
            opacity=0.5,
        )
        fig_px.update_traces(marker=dict(size=4))
        html_path = f"{out_dir}/html/clusters_{method}.html"
        fig_px.write_html(html_path)
        logger.info(f"Saved {html_path}")
    except ImportError:
        logger.warning("plotly not installed; skipping HTML output")


def plot_top_keywords_table(cluster_terms_path: str = "data/cluster_top_terms.csv",
                            cluster_map_path: str = "data/vector_clusters.csv",
                            out_dir: str = "output") -> None:
    """
    Generate a table image showing top TF-IDF keywords per schema dimension.
    """
    Path(f"{out_dir}/png").mkdir(parents=True, exist_ok=True)

    terms_df = pd.read_csv(cluster_terms_path, index_col="cluster_id")
    clusters_df = pd.read_csv(cluster_map_path)

    # Build cluster_id → schema_dimension map
    cid_to_dim = clusters_df.groupby("cluster_id")["schema_dimension"].first().to_dict()

    rows = []
    for cid, row in terms_df.iterrows():
        dim = cid_to_dim.get(cid, f"cluster_{cid}")
        keywords = ", ".join(str(t) for t in row.dropna().tolist()[:15])
        rows.append({"Dimension": dim, "Top Keywords": keywords})

    table_df = pd.DataFrame(rows).sort_values("Dimension")

    fig, ax = plt.subplots(figsize=(14, len(rows) * 1.2 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.auto_set_column_width([0, 1])
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    ax.set_title("Top Keywords per Schema Dimension (TF-IDF)",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    out_path = f"{out_dir}/png/top_keywords_table.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    plt.close()


def plot_sentiment_by_cluster(csv_path: str = "data/vector_clusters.csv",
                              out_dir: str = "output") -> None:
    """Stacked bar: positive/negative % per cluster."""
    Path(f"{out_dir}/png").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    sent = df.groupby(["schema_dimension", "sentiment_label"]).size().unstack(fill_value=0)
    sent["total"] = sent.sum(axis=1)
    for col in [c for c in sent.columns if c != "total"]:
        sent[col] = sent[col] / sent["total"] * 100
    sent = sent.drop(columns="total").sort_values("NEGATIVE", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sent[["POSITIVE", "NEGATIVE"]].plot(
        kind="barh", stacked=True, ax=ax,
        color=["#55A868", "#C44E52"], width=0.7
    )
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Sentiment Distribution by Cluster", fontsize=13, fontweight="bold")
    ax.legend(["Positive", "Negative"], loc="lower right")
    ax.axvline(50, color="white", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    out_path = f"{out_dir}/png/sentiment_by_cluster.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    plt.close()


def plot_engagement_by_cluster(csv_path: str = "data/vector_clusters.csv",
                               out_dir: str = "output") -> None:
    """Horizontal bar: avg votes per cluster."""
    Path(f"{out_dir}/png").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)
    avg = df.groupby("schema_dimension")["votes"].mean().sort_values()

    colors = [SCHEMA_COLORS.get(l, "#888888") for l in avg.index]
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(avg.index, avg.values, color=colors, height=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
    ax.set_xlabel("Average Votes per Comment")
    ax.set_title("Engagement (Avg Votes) by Cluster", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = f"{out_dir}/png/engagement_by_cluster.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    plt.close()


def plot_cluster_size_and_complexity(csv_path: str = "data/vector_clusters.csv",
                                     out_dir: str = "output") -> None:
    """Bubble chart: cluster size vs avg word count, bubble size = avg votes."""
    Path(f"{out_dir}/png").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df["votes"] = pd.to_numeric(df["votes"], errors="coerce").fillna(0)
    df["word_count"] = pd.to_numeric(df["word_count"], errors="coerce").fillna(0)
    stats = df.groupby("schema_dimension").agg(
        count=("text", "count"),
        avg_words=("word_count", "mean"),
        avg_votes=("votes", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(11, 8))
    for _, row in stats.iterrows():
        color = SCHEMA_COLORS.get(row["schema_dimension"], "#888888")
        ax.scatter(row["count"], row["avg_words"],
                   s=row["avg_votes"] * 80 + 40,
                   color=color, alpha=0.75, edgecolors="white", linewidth=0.8)
        ax.annotate(row["schema_dimension"],
                    (row["count"], row["avg_words"]),
                    fontsize=7.5, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")
    ax.set_xlabel("Cluster Size (# comments)")
    ax.set_ylabel("Avg Word Count per Comment")
    ax.set_title("Cluster Size vs Linguistic Complexity\n(bubble size = avg votes)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = f"{out_dir}/png/cluster_size_complexity.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {out_path}")
    plt.close()



def run(csv_path: str = "data/vector_clusters.csv",
        matrix: np.ndarray = None,
        method: str = "tsne") -> None:
    """
    Entry point. Pass pre-computed `matrix` from vector_pipeline,
    or it will re-build from the saved CSV (PCA only, no raw vectors stored).
    """
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_csv(csv_path)
    labels = df["schema_dimension"].values

    if matrix is None:
        logger.warning("No matrix passed; using PCA on numeric columns as fallback")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        matrix = df[num_cols].fillna(0).values
        method = "pca"

    plot_clusters(matrix, labels, method=method)
    plot_top_keywords_table()
    plot_sentiment_by_cluster(csv_path)
    plot_engagement_by_cluster(csv_path)
    plot_cluster_size_and_complexity(csv_path)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/vector_clusters.csv")
    p.add_argument("--method", default="tsne", choices=["tsne", "pca"])
    args = p.parse_args()
    run(args.csv, method=args.method)
