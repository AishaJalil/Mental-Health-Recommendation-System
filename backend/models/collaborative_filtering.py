import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class CollaborativeFilteringRecommender:
    """User-based and Item-based Collaborative Filtering."""

    def __init__(self, method="user"):
        self.method = method          # 'user' or 'item'
        self.rating_matrix = None
        self.similarity_df = None
        self.user_ids = []
        self.product_ids = []
        self.products_df = None
        self.ratings_df = None

    def fit(self, ratings_df, products_df):
        self.products_df = products_df
        self.ratings_df = ratings_df

        self.rating_matrix = ratings_df.pivot_table(
            index="user_id", columns="product_id", values="rating", fill_value=0
        )
        self.user_ids = self.rating_matrix.index.tolist()
        self.product_ids = self.rating_matrix.columns.tolist()

        if self.method == "user":
            sim = cosine_similarity(self.rating_matrix.values)
            self.similarity_df = pd.DataFrame(sim, index=self.user_ids, columns=self.user_ids)
        else:
            sim = cosine_similarity(self.rating_matrix.T.values)
            self.similarity_df = pd.DataFrame(sim, index=self.product_ids, columns=self.product_ids)

    def recommend(self, user_id, top_n=10, k_neighbors=15):
        if user_id not in self.user_ids:
            return []
        if self.method == "user":
            return self._user_based(user_id, top_n, k_neighbors)
        return self._item_based(user_id, top_n, k_neighbors)

    def _user_based(self, user_id, top_n, k):
        user_ratings = self.rating_matrix.loc[user_id]
        rated = set(user_ratings[user_ratings > 0].index)

        top_users = (
            self.similarity_df[user_id]
            .drop(user_id)
            .sort_values(ascending=False)
            .head(k)
        )

        predictions = {}
        for pid in self.product_ids:
            if pid in rated:
                continue
            num = denom = 0.0
            for sim_user, sim in top_users.items():
                r = self.rating_matrix.loc[sim_user, pid]
                if r > 0:
                    num += sim * r
                    denom += abs(sim)
            if denom > 0:
                predictions[pid] = num / denom

        top = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return self._build_results(top, "Users similar to you gave this {score:.1f}/5")

    def _item_based(self, user_id, top_n, k):
        user_ratings = self.rating_matrix.loc[user_id]
        rated = user_ratings[user_ratings > 0]
        unrated = set(self.product_ids) - set(rated.index)

        predictions = {}
        for pid in unrated:
            if pid not in self.similarity_df.columns:
                continue
            item_sims = self.similarity_df[pid]
            num = denom = 0.0
            for rated_pid, rating in rated.items():
                if rated_pid in item_sims.index:
                    s = item_sims[rated_pid]
                    num += s * rating
                    denom += abs(s)
            if denom > 0:
                predictions[pid] = num / denom

        top = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return self._build_results(top, "Similar to items you have enjoyed")

    def _build_results(self, top_pairs, reason_template):
        results = []
        for pid, score in top_pairs:
            row = self.products_df[self.products_df["product_id"] == pid]
            if row.empty:
                continue
            row = row.iloc[0]
            results.append({
                "product_id": pid,
                "name": row["name"],
                "category": row["category"],
                "price": float(row["price"]),
                "score": round(float(score), 3),
                "reason": reason_template.format(score=score),
            })
        return results

    def predict(self, user_id: str, product_id: str, k_neighbors: int = 15) -> float:
        """Return a predicted score without filtering already-seen items."""
        if user_id not in self.user_ids or product_id not in self.product_ids:
            return 0.0
        if self.method == "user":
            top_users = (
                self.similarity_df[user_id]
                .drop(user_id)
                .sort_values(ascending=False)
                .head(k_neighbors)
            )
            num = denom = 0.0
            for sim_user, sim in top_users.items():
                r = self.rating_matrix.loc[sim_user, product_id]
                if r > 0:
                    num   += sim * r
                    denom += abs(sim)
            return num / denom if denom > 0 else 0.0
        else:
            if product_id not in self.similarity_df.columns:
                return 0.0
            user_ratings = self.rating_matrix.loc[user_id]
            rated        = user_ratings[user_ratings > 0]
            item_sims    = self.similarity_df[product_id]
            num = denom  = 0.0
            for rated_pid, rating in rated.items():
                if rated_pid in item_sims.index and rated_pid != product_id:
                    s      = item_sims[rated_pid]
                    num   += s * rating
                    denom += abs(s)
            return num / denom if denom > 0 else 0.0

    def get_similar_products(self, product_id, top_n=5):
        if self.method != "item" or product_id not in self.similarity_df.columns:
            return []
        sims = (
            self.similarity_df[product_id]
            .drop(product_id)
            .sort_values(ascending=False)
            .head(top_n)
        )
        results = []
        for pid, score in sims.items():
            row = self.products_df[self.products_df["product_id"] == pid]
            if row.empty:
                continue
            row = row.iloc[0]
            results.append({
                "product_id": pid,
                "name": row["name"],
                "category": row["category"],
                "price": float(row["price"]),
                "similarity": round(float(score), 3),
            })
        return results
