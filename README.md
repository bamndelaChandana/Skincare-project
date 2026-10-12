# Personalized Skincare Recommendation System
**Academic Project — Enhanced Content-Based + Hybrid ML System**

## Improvements Over Base Paper

| Limitation (Base Paper) | Our Fix | Justification |
|---|---|---|
| Binary one-hot encoding ignores ingredient importance | **TF-IDF with sublinear_tf** | Active ingredients rank higher; "water" no longer dominates |
| Synonym blindness ("Niacinamide" ≠ "Vitamin B3") | **Pre-trained transformer embeddings** (all-MiniLM-L6-v2) | Model trained on biomedical text already captures synonyms |
| t-SNE: non-deterministic, expensive, no new-point support | **PCA → UMAP pipeline** | Deterministic, faster, can transform new products |
| Static content-based system | **Hybrid: content + user profile** | User profile = mean of liked embeddings → real personalisation |
| No empirical metric justification | **Alpha grid search + method comparison** | α=0.7 proven optimal; improvement % shown vs baseline |

## Project Structure

```
skincare-recommender/
├── data/cosmetic_p.csv
├── models/               (auto-created, stores feedback.json)
├── src/
│   ├── __init__.py
│   ├── preprocess.py     TF-IDF feature extraction
│   ├── embeddings.py     Pre-trained transformer embeddings
│   ├── features.py       Fuse TF-IDF + embeddings
│   ├── dimensionality.py PCA + UMAP reduction
│   ├── recommender.py    Weighted similarity + hybrid scoring
│   ├── feedback.py       User profile + feedback persistence
│   └── evaluate.py       Alpha grid search + method comparison
├── app.py                Streamlit UI
├── requirements.txt
└── README.md
```

## Setup & Run

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

## Key Formula

```
final_score = α × cosine_similarity + (1−α) × euclidean_similarity
            where α = 0.7  (empirically determined via grid search)

hybrid_score = 0.6 × content_score + 0.4 × profile_score
             where profile = mean(embeddings of liked products)
```