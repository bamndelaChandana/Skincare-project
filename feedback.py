"""
feedback.py
-----------
Persistent user feedback + user profile computation.

This is what makes the system genuinely HYBRID (not just content-based):

  1. User clicks 👍 on a product.
  2. That product's embedding is saved in feedback.json.
  3. build_user_profile() averages all liked embeddings into a single
     'taste vector' in the same embedding space.
  4. hybrid() in recommender.py blends content similarity with similarity
     to this taste vector.

Result: over time, the system steers towards products the user has
explicitly liked — a real adaptive feedback loop.

Academic note: this is a lightweight form of implicit collaborative
filtering without requiring multiple users. The user profile acts as
a personalised query vector in the latent ingredient space.
"""

import json
import os
import numpy as np

FEEDBACK_FILE = "models/feedback.json"


# ──────────────────────────────────────────────
#  Persistence helpers
# ──────────────────────────────────────────────

def _ensure_dir():
    os.makedirs("models", exist_ok=True)


def load_feedback() -> dict:
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {}


def save_feedback(feedback: dict):
    _ensure_dir()
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f, indent=2)


# ──────────────────────────────────────────────
#  Record a single like / dislike
# ──────────────────────────────────────────────

def record_feedback(product_name: str, liked: bool, skin_type: str = ""):
    """
    Adds one like or dislike for product_name.
    Also records which skin type was active — useful for future analysis.
    """
    feedback = load_feedback()
    if product_name not in feedback:
        feedback[product_name] = {"likes": 0, "dislikes": 0, "skin_types": []}
    if liked:
        feedback[product_name]["likes"] += 1
    else:
        feedback[product_name]["dislikes"] += 1
    if skin_type and skin_type not in feedback[product_name]["skin_types"]:
        feedback[product_name]["skin_types"].append(skin_type)
    save_feedback(feedback)


# ──────────────────────────────────────────────
#  Build user profile from liked products
# ──────────────────────────────────────────────

def get_liked_product_names() -> list:
    """Returns list of product names that have at least 1 more like than dislike."""
    feedback = load_feedback()
    liked = [
        name for name, data in feedback.items()
        if data.get("likes", 0) > data.get("dislikes", 0)
    ]
    return liked


def build_user_profile(df, embedding_matrix: np.ndarray):
    """
    User profile = mean embedding of all net-liked products.

    Returns None if no feedback has been given yet.

    This vector lives in the same 384-dimensional space as product embeddings,
    so cosine_similarity(profile, product_embedding) is directly meaningful.
    """
    liked_names = get_liked_product_names()
    if not liked_names:
        return None

    indices = []
    for name in liked_names:
        match = df.index[df["name"] == name].tolist()
        if match:
            indices.append(match[0])

    if not indices:
        return None

    profile = np.mean(embedding_matrix[indices], axis=0)
    # L2 normalise so it behaves like the other unit-length embeddings
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    return profile


# ──────────────────────────────────────────────
#  Simple score boost for display re-ranking
# ──────────────────────────────────────────────

def apply_feedback_boost(recommendations_df):
    """
    Nudges similarity_score by ±0.01 per net like/dislike.
    Used only for display ordering after hybrid() already scored products.
    """
    feedback = load_feedback()

    def boost(row):
        fb = feedback.get(row["name"], {})
        delta = 0.01 * fb.get("likes", 0) - 0.01 * fb.get("dislikes", 0)
        return row["similarity_score"] + delta

    recommendations_df = recommendations_df.copy()
    recommendations_df["adjusted_score"] = recommendations_df.apply(boost, axis=1)
    return recommendations_df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
#  Stats for display in UI
# ──────────────────────────────────────────────

def get_feedback_stats() -> dict:
    feedback = load_feedback()
    total_liked    = sum(1 for v in feedback.values() if v.get("likes", 0) > 0)
    total_disliked = sum(1 for v in feedback.values() if v.get("dislikes", 0) > 0)
    return {
        "total_interactions": len(feedback),
        "liked": total_liked,
        "disliked": total_disliked,
    }