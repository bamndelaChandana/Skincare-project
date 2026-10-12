"""
dimensionality.py
-----------------
Two-stage dimensionality reduction: PCA → UMAP.

Why this is better than t-SNE (base paper):
  ┌──────────────┬────────────────────────────────┬──────────────────────────┐
  │ Property     │ t-SNE (base paper)             │ PCA + UMAP (this work)   │
  ├──────────────┼────────────────────────────────┼──────────────────────────┤
  │ Determinism  │ Non-deterministic (changes     │ Deterministic with       │
  │              │ every run, random_state alone  │ random_state=42          │
  │              │ doesn't fully fix layout)      │                          │
  ├──────────────┼────────────────────────────────┼──────────────────────────┤
  │ New products │ Full retraining required       │ PCA.transform() can      │
  │              │                                │ project new points       │
  ├──────────────┼────────────────────────────────┼──────────────────────────┤
  │ Speed        │ O(n²) — slow on large data     │ PCA pre-reduces to 50D;  │
  │              │                                │ UMAP is O(n log n)       │
  ├──────────────┼────────────────────────────────┼──────────────────────────┤
  │ Structure    │ Preserves local clusters only  │ PCA preserves global,    │
  │              │                                │ UMAP preserves local     │
  └──────────────┴────────────────────────────────┴──────────────────────────┘

Pipeline:
  raw features (3100d)  →  PCA (50d, noise removal)  →  UMAP (2d, visualise)
"""

import numpy as np
from sklearn.decomposition import PCA
import umap


def reduce_dimensions(
    matrix,
    n_pca_components: int = 50,
    random_state: int = 42,
):
    """
    Parameters
    ----------
    matrix          : sparse or dense array (n_products, n_features)
    n_pca_components: intermediate PCA dimensions before UMAP
    random_state    : fixed seed for full reproducibility

    Returns
    -------
    umap_coords : np.ndarray (n_products, 2)  — the 2D map
    pca         : fitted PCA object
    reducer     : fitted UMAP object
    """
    # --- dense conversion (PCA needs dense) ---
    if hasattr(matrix, "toarray"):
        dense = matrix.toarray()
    else:
        dense = np.array(matrix)

    # --- Stage 1: PCA ---
    n_pca = min(n_pca_components, dense.shape[1] - 1, dense.shape[0] - 1)
    print(f"[dimensionality] PCA: {dense.shape[1]}d → {n_pca}d")
    pca = PCA(n_components=n_pca, random_state=random_state)
    pca_result = pca.fit_transform(dense)

    # --- Stage 2: UMAP ---
    print(f"[dimensionality] UMAP: {n_pca}d → 2d")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,     # local neighbourhood size
        min_dist=0.1,       # how tightly points are packed
        metric="cosine",    # cosine is better for text/embedding vectors
        random_state=random_state,
    )
    umap_coords = reducer.fit_transform(pca_result)
    print("[dimensionality] Done.")
    return umap_coords, pca, reducer