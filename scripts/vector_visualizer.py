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
    "Technical Aesthetics": "#4C72B0",
    "Ontological Evaluation": "#DD8452",
    "Post-human Poetics": "#55A868",
    "Affective Resonance": "#C44E52",
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


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/vector_clusters.csv")
    p.add_argument("--method", default="tsne", choices=["tsne", "pca"])
    args = p.parse_args()
    run(args.csv, method=args.method)
