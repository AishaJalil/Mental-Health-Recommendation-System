"""
Content-Based Filtering
Primary:  Sentence-BERT embeddings (all-MiniLM-L6-v2, 384-dim)
Fallback: TF-IDF + category one-hot + normalised price  (if sentence-transformers not installed)
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False


class ContentBasedRecommender:
    # Class-level model cache — loaded once, shared across instances
    _sbert_instance = None

    def __init__(self, sbert_model="all-MiniLM-L6-v2"):
        self.sbert_model_name = sbert_model
        self.products_df      = None
        self.ratings_df       = None
        self.product_ids      = []

        # SBERT mode
        self._embeddings  = None   # (n_items, 384) float32, L2-normalised
        self._sim_matrix  = None   # (n_items, n_items) float32

        # TF-IDF fallback mode
        self._tfidf_mode     = False
        self._tfidf_features = None
        self._similarity_df  = None

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, products_df, ratings_df):
        self.products_df = products_df.copy()
        self.ratings_df  = ratings_df
        self.product_ids = products_df["product_id"].tolist()

        if _SBERT_OK:
            self._fit_sbert(products_df)
        else:
            print("  [CBF] sentence-transformers not installed — using TF-IDF fallback")
            print("        Install: pip install sentence-transformers")
            self._fit_tfidf(products_df)

    def _fit_sbert(self, products_df):
        if ContentBasedRecommender._sbert_instance is None:
            print(f"  [CBF] Loading Sentence-BERT ({self.sbert_model_name}) — first run downloads ~80 MB …")
            ContentBasedRecommender._sbert_instance = SentenceTransformer(self.sbert_model_name)

        sbert = ContentBasedRecommender._sbert_instance

        # Rich text per activity: name + category + tags + clinical description
        texts = (
            "Activity: "  + products_df["name"].fillna("") + ". "
            + "Category: " + products_df["category"].fillna("") + ". "
            + "Tags: "     + products_df["tags"].fillna("") + ". "
            + products_df["description"].fillna("")
        ).tolist()

        print(f"  [CBF] Encoding {len(texts)} activities with Sentence-BERT …")
        emb   = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # L2-normalise so dot product = cosine similarity
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        self._embeddings = (emb / np.maximum(norms, 1e-9)).astype(np.float32)
        self._sim_matrix = (self._embeddings @ self._embeddings.T).astype(np.float32)
        self._tfidf_mode = False
        print("  [CBF] Sentence-BERT content model ready.")

    def _fit_tfidf(self, products_df):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import MinMaxScaler
        from scipy.sparse import hstack, csr_matrix

        text = (
            products_df["tags"] + " "
            + products_df["category"].str.lower() + " "
            + products_df["description"].str.lower()
        )
        vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
        tfidf = vectorizer.fit_transform(text)

        prices      = products_df["price"].values.reshape(-1, 1)
        norm_prices = csr_matrix(MinMaxScaler().fit_transform(prices) * 0.5)
        cat_dummies = pd.get_dummies(products_df["category"], prefix="cat")
        cat_sparse  = csr_matrix(cat_dummies.values.astype(float) * 2.0)

        self._tfidf_features = hstack([tfidf, cat_sparse, norm_prices])
        sim = cosine_similarity(self._tfidf_features)
        self._similarity_df  = pd.DataFrame(sim, index=self.product_ids, columns=self.product_ids)
        self._tfidf_mode     = True

    # ── User profile ──────────────────────────────────────────────────────────

    def _build_user_profile(self, user_id):
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            return None

        if self._tfidf_mode:
            profile = None
            total_w = 0.0
            for _, row in user_ratings.iterrows():
                pid = row["product_id"]
                if pid not in self.product_ids:
                    continue
                idx  = self.product_ids.index(pid)
                w    = float(row["rating"]) - 2.5
                feat = self._tfidf_features[idx]
                profile = feat if profile is None else profile + w * feat
                total_w += abs(w)
            if total_w > 0 and profile is not None:
                return profile / total_w
            return profile
        else:
            dim     = self._embeddings.shape[1]
            profile = np.zeros(dim, dtype=np.float32)
            total_w = 0.0
            for _, row in user_ratings.iterrows():
                pid = row["product_id"]
                if pid not in self.product_ids:
                    continue
                idx = self.product_ids.index(pid)
                w   = float(row["rating"]) - 2.5
                profile += w * self._embeddings[idx]
                total_w += abs(w)
            if total_w > 0:
                profile = profile / total_w
            return profile

    # ── Recommend ─────────────────────────────────────────────────────────────

    def recommend(self, user_id, top_n=10):
        profile = self._build_user_profile(user_id)
        if profile is None:
            return []

        rated = set(
            self.ratings_df[self.ratings_df["user_id"] == user_id]["product_id"].tolist()
        )

        if self._tfidf_mode:
            scores = cosine_similarity(profile, self._tfidf_features)[0]
        else:
            p_norm = profile / (np.linalg.norm(profile) + 1e-9)
            scores = (self._embeddings @ p_norm).tolist()

        candidates = []
        for idx, score in enumerate(scores):
            pid = self.product_ids[idx]
            if pid in rated:
                continue
            row    = self.products_df[self.products_df["product_id"] == pid].iloc[0]
            reason = (
                f"Semantically matched to your activity profile ({row['category']})"
                if not self._tfidf_mode
                else f"Matches your taste profile in {row['category']}"
            )
            candidates.append({
                "product_id": pid,
                "name":       row["name"],
                "category":   row["category"],
                "price":      float(row["price"]),
                "score":      round(float(score), 3),
                "reason":     reason,
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    def predict(self, user_id: str, product_id: str) -> float:
        """Return a relevance score for user–item without filtering already-seen items."""
        profile = self._build_user_profile(user_id)
        if profile is None or product_id not in self.product_ids:
            return 0.0
        idx = self.product_ids.index(product_id)
        if self._tfidf_mode:
            score = float(cosine_similarity(profile, self._tfidf_features[idx: idx + 1])[0][0])
        else:
            p_norm = profile / (np.linalg.norm(profile) + 1e-9)
            score  = float(self._embeddings[idx] @ p_norm)
        return score

    # ── Similar products ──────────────────────────────────────────────────────

    def get_similar_products(self, product_id, top_n=5):
        if product_id not in self.product_ids:
            return []

        idx = self.product_ids.index(product_id)

        if self._tfidf_mode:
            sims = (
                self._similarity_df[product_id]
                .drop(product_id)
                .sort_values(ascending=False)
                .head(top_n)
            )
            sim_pairs = list(sims.items())
        else:
            row_sims  = self._sim_matrix[idx].tolist()
            sim_pairs = sorted(
                [(self.product_ids[i], s)
                 for i, s in enumerate(row_sims)
                 if self.product_ids[i] != product_id],
                key=lambda x: x[1], reverse=True,
            )[:top_n]

        results = []
        for pid, score in sim_pairs:
            r = self.products_df[self.products_df["product_id"] == pid]
            if r.empty:
                continue
            r = r.iloc[0]
            results.append({
                "product_id": pid,
                "name":       r["name"],
                "category":   r.get("category", ""),
                "similarity": round(float(score), 4),
            })
        return results
