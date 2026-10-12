"""
embeddings.py
-------------
Builds semantic embeddings using a PRE-TRAINED transformer model.

Why this beats Word2Vec trained from scratch:
  - Word2Vec needs a large corpus to learn meaning.  1,472 products is far
    too small — the embeddings would be noise.
  - 'all-MiniLM-L6-v2' was trained on 1 billion+ sentence pairs.  It already
    knows that "niacinamide" and "vitamin b3" are the same compound, that
    "retinol" and "retinoic acid" are related, etc.
  - This directly fixes the base paper's "limited semantic understanding"
    limitation without needing any external labelled data.

Model choice justification (for viva):
  'all-MiniLM-L6-v2' — small (80 MB), fast, strong on short text.
  Academic upgrade: swap MODEL_NAME for
  'pritamdeka/S-PubMedBert-MS-MARCO' for biomedical specialisation.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def build_embedding_matrix(df, model_name: str = MODEL_NAME) -> tuple:
    """
    Parameters
    ----------
    df         : DataFrame with 'clean_ingredients' column (from preprocess.py)
    model_name : HuggingFace model ID

    Returns
    -------
    embeddings : np.ndarray  shape (n_products, embedding_dim)
    model      : loaded SentenceTransformer (reuse to embed new queries)
    """
    model = SentenceTransformer(model_name)

    texts = df["clean_ingredients"].tolist()

    print(f"[embeddings] Encoding {len(texts)} products with '{model_name}'...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit-length vectors: cosine sim = dot product
        convert_to_numpy=True,
    )
    print(f"[embeddings] Done. Shape: {embeddings.shape}")
    return embeddings, model