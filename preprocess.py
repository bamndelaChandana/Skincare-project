"""
preprocess.py
-------------
Loads the CSV, cleans ingredient text, and builds the TF-IDF matrix.

Key design choices vs base paper:
  - sublinear_tf=True  : log-normalises term frequency so "water" (in 90% of
                         products) doesn't dominate the vector.
  - ngram_range=(1,2)  : treats "hyaluronic acid" as ONE feature, not two
                         unrelated words.
  - max_features=3000  : keeps the vocabulary manageable without losing
                         meaningful active ingredients.
"""

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(path: str = "data/cosmetic_p.csv") -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df["ingredients"] = df["ingredients"].fillna("")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0.0)
    # make sure skin-type columns are integer 0/1
    for col in ["Combination", "Dry", "Normal", "Oily", "Sensitive"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def clean_ingredients(text: str) -> str:
    """Lower-case, strip special chars, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s,]", " ", text)   # keep commas as separators
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_tfidf(df: pd.DataFrame):
    """
    Returns
    -------
    tfidf_matrix : sparse matrix  shape (n_products, n_features)
    vectorizer   : fitted TfidfVectorizer
    df           : dataframe with new 'clean_ingredients' column added
    """
    df = df.copy()
    df["clean_ingredients"] = df["ingredients"].apply(clean_ingredients)

    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        token_pattern=r"[a-z][a-z0-9\s\-]{1,}",
    )
    tfidf_matrix = vectorizer.fit_transform(df["clean_ingredients"])
    return tfidf_matrix, vectorizer, df