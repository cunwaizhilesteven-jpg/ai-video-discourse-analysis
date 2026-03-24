"""
Vector Pipeline - Bottom-up Semantic Analysis
Implements Word2Vec clustering + SBERT for cultural analytics.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ── 1. Preprocessing ──────────────────────────────────────────────────────────

def load_and_preprocess(csv_path: str = "data/merged_data.csv") -> pd.DataFrame:
    """Load CSV, de-identify authors, tokenize + lemmatize with spaCy."""
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    df = pd.read_csv(csv_path)
    df["author"] = "user_" + pd.factorize(df["author"])[0].astype(str)  # de-identify

    texts = df["text"].fillna("").astype(str).tolist()
    tokens_list = []
    for doc in nlp.pipe(texts, batch_size=256):
        tokens = [
            t.lemma_.lower() for t in doc
            if t.is_alpha and not t.is_stop and len(t) > 2
        ]
        tokens_list.append(tokens)

    df["tokens"] = tokens_list
    logger.info(f"Preprocessed {len(df)} comments")
    return df


# ── 2. Word2Vec Vectorization ─────────────────────────────────────────────────

def train_word2vec(tokens_list: list, vector_size: int = 100, min_count: int = 5) -> Word2Vec:
    model = Word2Vec(
        sentences=tokens_list,
        vector_size=vector_size,
        window=5,
        min_count=min_count,
        workers=4,
        epochs=10,
    )
    logger.info(f"Word2Vec vocab size: {len(model.wv)}")
    return model


def comment_vector(tokens: list, wv) -> np.ndarray:
    """Mean-pool token vectors for a comment."""
    vecs = [wv[t] for t in tokens if t in wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(wv.vector_size)


def build_comment_matrix(df: pd.DataFrame, wv) -> np.ndarray:
    matrix = np.vstack([comment_vector(t, wv) for t in df["tokens"]])
    return normalize(matrix)


# ── 3. Clustering ─────────────────────────────────────────────────────────────

def cluster_comments(matrix: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)
    logger.info(f"Clustered into {n_clusters} groups")
    return labels, km


def top_terms_per_cluster(df: pd.DataFrame, labels: np.ndarray, wv, top_n: int = 20) -> dict:
    """
    For each cluster, extract top-N terms using TF-IDF across cluster documents.
    Falls back to frequency if TF-IDF fails.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    result = {}
    cluster_docs = {}
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        # Join tokens into pseudo-documents per comment
        docs = [" ".join(t for t in tokens if t in wv)
                for tokens in df.loc[mask, "tokens"]]
        docs = [d for d in docs if d.strip()]
        cluster_docs[cluster_id] = docs

    try:
        all_docs = []
        cluster_order = []
        for cid, docs in cluster_docs.items():
            all_docs.append(" ".join(docs))  # one mega-doc per cluster
            cluster_order.append(cid)
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 1))
        tfidf_matrix = tfidf.fit_transform(all_docs)
        feature_names = tfidf.get_feature_names_out()
        for i, cid in enumerate(cluster_order):
            row = tfidf_matrix[i].toarray().flatten()
            top_idx = row.argsort()[::-1][:top_n]
            result[cid] = [feature_names[j] for j in top_idx]
    except Exception:
        from collections import Counter
        for cluster_id in sorted(set(labels)):
            mask = labels == cluster_id
            all_tokens = [t for tokens in df.loc[mask, "tokens"] for t in tokens if t in wv]
            freq = Counter(all_tokens)
            result[cluster_id] = [term for term, _ in freq.most_common(top_n)]
    return result


# ── 4. Cosine Similarity between high-frequency terms ────────────────────────

def term_similarity(terms: list[str], wv) -> pd.DataFrame:
    """Return pairwise cosine similarity matrix for given terms."""
    valid = [t for t in terms if t in wv]
    vecs = np.vstack([wv[t] for t in valid])
    sim = cosine_similarity(vecs)
    return pd.DataFrame(sim, index=valid, columns=valid)


# ── 4b. Textual Anchor mapping ────────────────────────────────────────────────

# Theoretical definitions used as textual anchors (no fixed seed words).
# Each anchor phrase is vectorized via mean-pool of its constituent word vectors.
TEXTUAL_ANCHORS = {
    "Technical Aesthetics": "glitch render error uncanny anomaly automated beauty visual artifact",
    "Ontological Evaluation": "mind intentionality soul agency credit consciousness authentic artificial",
    "Post-human Poetics": "rhythm flow synchronization structure beauty audio visual posthuman",
    "Affective Resonance": "moving profound inspiring emotional shock autonomous feel touch",
}


def _anchor_vector(phrase: str, wv) -> np.ndarray:
    """Mean-pool word vectors for an anchor phrase."""
    tokens = [t for t in phrase.lower().split() if t in wv]
    if not tokens:
        return np.zeros(wv.vector_size)
    return np.mean([wv[t] for t in tokens], axis=0)


def map_clusters_by_anchor_similarity(cluster_centroids: np.ndarray, wv) -> dict:
    """
    Compare each cluster centroid to all 4 textual anchor vectors via cosine
    similarity. Return {cluster_id: dimension_name} using Hungarian-style
    greedy assignment (highest similarity wins, each dimension used once).
    Also returns the full similarity matrix as a DataFrame for inspection.
    """
    dims = list(TEXTUAL_ANCHORS.keys())
    anchor_vecs = np.vstack([_anchor_vector(TEXTUAL_ANCHORS[d], wv) for d in dims])
    anchor_vecs = normalize(anchor_vecs)
    centroids_normed = normalize(cluster_centroids)

    sim_matrix = cosine_similarity(centroids_normed, anchor_vecs)  # (n_clusters, 4)
    sim_df = pd.DataFrame(sim_matrix,
                          index=[f"cluster_{i}" for i in range(len(cluster_centroids))],
                          columns=dims)

    # Greedy assignment: pick best unassigned dimension per cluster
    assigned_dims = set()
    mapping = {}
    for _ in range(len(cluster_centroids)):
        best_score = -1
        best_cluster = best_dim = None
        for cid in range(len(cluster_centroids)):
            if cid in mapping:
                continue
            for j, dim in enumerate(dims):
                if dim in assigned_dims:
                    continue
                if sim_matrix[cid, j] > best_score:
                    best_score = sim_matrix[cid, j]
                    best_cluster, best_dim = cid, dim
        mapping[best_cluster] = best_dim
        assigned_dims.add(best_dim)

    logger.info("Anchor similarity matrix:\n" + sim_df.to_string())
    logger.info(f"Cluster → dimension mapping: {mapping}")
    return mapping, sim_df


def find_semantic_overlap(df: pd.DataFrame, matrix: np.ndarray, wv,
                          dim_a: str = "Ontological Evaluation",
                          dim_b: str = "Post-human Poetics",
                          threshold: float = 0.35) -> pd.DataFrame:
    """
    Extract comments with cosine similarity > threshold to BOTH dim_a and dim_b.
    Returns de-identified DataFrame for Phase 2 close reading.
    """
    anchor_a = normalize(_anchor_vector(TEXTUAL_ANCHORS[dim_a], wv).reshape(1, -1))
    anchor_b = normalize(_anchor_vector(TEXTUAL_ANCHORS[dim_b], wv).reshape(1, -1))
    mat_normed = normalize(matrix)

    sim_a = cosine_similarity(mat_normed, anchor_a).flatten()
    sim_b = cosine_similarity(mat_normed, anchor_b).flatten()

    mask = (sim_a > threshold) & (sim_b > threshold)
    overlap_df = df[mask].copy()
    overlap_df["sim_" + dim_a.replace(" ", "_")] = sim_a[mask]
    overlap_df["sim_" + dim_b.replace(" ", "_")] = sim_b[mask]
    # Ensure de-identification
    if "author" in overlap_df.columns:
        overlap_df["author"] = "user_" + pd.factorize(overlap_df["author"])[0].astype(str)
    logger.info(f"Semantic overlap ({dim_a} ∩ {dim_b}): {mask.sum()} comments")
    return overlap_df.drop(columns=["tokens"], errors="ignore")


# ── 5. Cluster → Schema Mapping ───────────────────────────────────────────────

def map_clusters_to_schema(labels: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Map cluster IDs to schema dimension names using the provided mapping dict
    {cluster_id: dimension_name} from map_clusters_by_anchor_similarity.
    """
    return np.array([mapping[l] for l in labels])


# ── 6. Stage-2 SBERT preparation ─────────────────────────────────────────────

def encode_with_sbert(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Encode comments with Sentence-BERT for comment-level vectorization.
    Call this after Stage 1 to refine cluster assignments.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                               convert_to_numpy=True)
    return normalize(embeddings)


# ── 7. Main pipeline ──────────────────────────────────────────────────────────

def run(csv_path: str = "data/merged_data.csv",
        n_clusters: int = 4,
        use_sbert: bool = False) -> pd.DataFrame:

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    df = load_and_preprocess(csv_path)

    w2v = train_word2vec(df["tokens"].tolist())
    wv = w2v.wv

    matrix = build_comment_matrix(df, wv)

    if use_sbert:
        logger.info("Stage 2: encoding with SBERT...")
        matrix = encode_with_sbert(df["text"].fillna("").tolist())

    labels, km = cluster_comments(matrix, n_clusters)
    df["cluster_id"] = labels

    cluster_terms = top_terms_per_cluster(df, labels, wv)
    for cid, terms in cluster_terms.items():
        logger.info(f"Cluster {cid} top terms: {terms}")

    # Anchor-based mapping: compare cluster centroids to theoretical anchor vectors
    mapping, sim_df = map_clusters_by_anchor_similarity(km.cluster_centers_, wv)
    sim_df.to_csv("data/anchor_similarity.csv")
    logger.info("Saved anchor_similarity.csv")

    df["schema_dimension"] = map_clusters_to_schema(labels, mapping)

    out = Path("data/vector_clusters.csv")
    df.drop(columns=["tokens"]).to_csv(out, index=False, encoding="utf-8")
    logger.info(f"Saved to {out}")

    # Save top terms for manual inspection
    terms_df = pd.DataFrame.from_dict(cluster_terms, orient="index")
    terms_df.index.name = "cluster_id"
    terms_df.to_csv("data/cluster_top_terms.csv")
    logger.info("Saved cluster_top_terms.csv")

    # Extract semantic overlap: Ontological Evaluation ∩ Post-human Poetics
    overlap_df = find_semantic_overlap(df, matrix, wv)
    overlap_df.to_csv("data/semantic_overlap.csv", index=False, encoding="utf-8")
    logger.info("Saved semantic_overlap.csv")

    return df, wv, cluster_terms, km


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/merged_data.csv")
    p.add_argument("--clusters", type=int, default=4)
    p.add_argument("--sbert", action="store_true", help="Use SBERT for Stage 2")
    args = p.parse_args()
    run(args.csv, args.clusters, args.sbert)
