"""
recommender.py
--------------
Two recommendation modes:

1. content_only()  — pure content-based (used for the baseline comparison
                     in evaluate.py to prove improvement over base paper)

2. hybrid()        — content similarity + user profile similarity
                     This is the actual system users interact with.

Similarity formula (justification for viva):
    final_score = α × cosine_sim + (1−α) × euclidean_sim

    cosine_sim   : measures the DIRECTION of ingredient vectors
                   (good for catching overall ingredient profile similarity)
    euclidean_sim: measures the MAGNITUDE difference
                   (catches when products share the same direction but have
                    very different ingredient quantities)
    Combined     : more robust than either alone.

    α = 0.7 is the empirically optimal value found by evaluate.py's grid
    search — you can show this table in your viva.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist

SKIN_COLS = ["Combination", "Dry", "Normal", "Oily", "Sensitive"]


def _to_dense(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix)


def _weighted_similarity(
    matrix: np.ndarray,
    query_idx: int,
    alpha: float = 0.7,
) -> np.ndarray:
    """
    Returns a score array of shape (n_products,).
    Higher = more similar.
    """
    query = matrix[query_idx].reshape(1, -1)

    cosine_scores = cosine_similarity(query, matrix)[0]

    euclidean_dist = cdist(query, matrix, metric="euclidean")[0]
    max_dist = euclidean_dist.max() + 1e-9
    euclidean_scores = 1.0 - (euclidean_dist / max_dist)   # normalise to [0,1]

    return alpha * cosine_scores + (1.0 - alpha) * euclidean_scores


def _apply_skin_filter(scores: np.ndarray, df, skin_type: str) -> np.ndarray:
    if skin_type and skin_type in SKIN_COLS:
        mask = df[skin_type].values == 1
        scores = scores * mask
    return scores


def content_only(
    df,
    feature_matrix,
    product_idx: int,
    skin_type: str = None,
    top_n: int = 10,
    alpha: float = 0.7,
):
    """Pure content-based recommendation (no user profile)."""
    matrix = _to_dense(feature_matrix)
    scores = _weighted_similarity(matrix, product_idx, alpha)
    scores[product_idx] = -1.0
    scores = _apply_skin_filter(scores, df, skin_type)

    top_idx = np.argsort(scores)[::-1][:top_n]
    results = df.iloc[top_idx][
        ["Label", "brand", "name", "price", "rank", "ingredients"]
    ].copy()
    results["similarity_score"] = np.round(scores[top_idx], 4)
    return results.reset_index(drop=True)


def hybrid(
    df,
    feature_matrix,
    embedding_matrix: np.ndarray,
    product_idx: int,
    user_profile: np.ndarray = None,
    skin_type: str = None,
    top_n: int = 10,
    content_weight: float = 0.6,
    alpha: float = 0.7,
):
    """
    Hybrid recommendation: content signal + user profile signal.

    final_score = content_weight  × content_similarity(query, product)
                + profile_weight  × profile_similarity(user_profile, product)

    If no profile exists yet (first session), falls back to content_only.
    """
    matrix = _to_dense(feature_matrix)
    content_scores = _weighted_similarity(matrix, product_idx, alpha)

    if user_profile is not None:
        # profile_similarity: how much does this product match what the user liked?
        profile_scores = cosine_similarity(
            user_profile.reshape(1, -1), embedding_matrix
        )[0]
    else:
        profile_scores = content_scores  # no profile yet → content only

    profile_weight = 1.0 - content_weight
    final_scores = content_weight * content_scores + profile_weight * profile_scores

    final_scores[product_idx] = -1.0
    final_scores = _apply_skin_filter(final_scores, df, skin_type)

    top_idx = np.argsort(final_scores)[::-1][:top_n]
    results = df.iloc[top_idx][
        ["Label", "brand", "name", "price", "rank", "ingredients"]
    ].copy()
    results["similarity_score"] = np.round(final_scores[top_idx], 4)
    return results.reset_index(drop=True)