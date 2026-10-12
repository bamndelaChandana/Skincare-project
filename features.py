"""
features.py
-----------
Fuses the TF-IDF matrix (precise, sparse) with transformer embeddings
(semantic, dense) into one unified feature matrix.

Design rationale:
  TF-IDF alone : knows WHICH ingredients are rare/important but not their
                 meaning or relationships.
  Embeddings   : understand meaning and synonyms but are less precise about
                 exact ingredient counts.
  Combined     : gets both — precise AND semantically aware.

Weighting (tfidf_weight=0.6):
  TF-IDF slightly dominates because the task is ingredient-similarity, where
  exact ingredient presence matters more than semantics.
  The optimal weight is confirmed empirically by evaluate.py.
"""

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import normalize


def combine_features(
    tfidf_matrix,
    embedding_matrix: np.ndarray,
    tfidf_weight: float = 0.6,
):
    """
    Parameters
    ----------
    tfidf_matrix      : sparse matrix (n, f1) from build_tfidf()
    embedding_matrix  : dense array  (n, f2) from build_embedding_matrix()
    tfidf_weight      : float in [0,1]; weight given to TF-IDF component

    Returns
    -------
    combined : sparse matrix (n, f1+f2), L2-normalised column-wise
    """
    emb_weight = 1.0 - tfidf_weight

    # L2-normalise both so they're on the same scale before weighting
    tfidf_norm = normalize(tfidf_matrix, norm="l2")
    emb_norm   = normalize(csr_matrix(embedding_matrix), norm="l2")

    combined = hstack(
        [tfidf_norm * tfidf_weight, emb_norm * emb_weight],
        format="csr",
    )
    return combined