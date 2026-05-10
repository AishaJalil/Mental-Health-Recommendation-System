import numpy as np
import pandas as pd


class MatrixFactorizationRecommender:
    """
    SVD-style Matrix Factorization trained with SGD.
    Learns latent user/item factors plus bias terms.
    """

    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.global_mean = 0.0
        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.user_to_idx = {}
        self.item_to_idx = {}
        self.users = []
        self.items = []
        self.products_df = None
        self.ratings_df = None

    def fit(self, ratings_df, products_df):
        self.products_df = products_df
        self.ratings_df = ratings_df

        self.users = ratings_df["user_id"].unique().tolist()
        self.items = ratings_df["product_id"].unique().tolist()
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.item_to_idx = {it: i for i, it in enumerate(self.items)}

        n_u, n_i = len(self.users), len(self.items)
        self.global_mean = float(ratings_df["rating"].mean())

        np.random.seed(42)
        scale = 0.1
        self.user_factors = np.random.normal(0, scale, (n_u, self.n_factors))
        self.item_factors = np.random.normal(0, scale, (n_i, self.n_factors))
        self.user_biases = np.zeros(n_u)
        self.item_biases = np.zeros(n_i)

        data = ratings_df[["user_id", "product_id", "rating"]].values.tolist()

        for _ in range(self.n_epochs):
            np.random.shuffle(data)
            for user_id, item_id, rating in data:
                if user_id not in self.user_to_idx or item_id not in self.item_to_idx:
                    continue
                u = self.user_to_idx[user_id]
                i = self.item_to_idx[item_id]

                pred = self._predict_idx(u, i)
                err = float(rating) - pred

                self.user_biases[u] += self.lr * (err - self.reg * self.user_biases[u])
                self.item_biases[i] += self.lr * (err - self.reg * self.item_biases[i])

                uf_copy = self.user_factors[u].copy()
                self.user_factors[u] += self.lr * (err * self.item_factors[i] - self.reg * self.user_factors[u])
                self.item_factors[i] += self.lr * (err * uf_copy - self.reg * self.item_factors[i])

    def _predict_idx(self, u, i):
        pred = (
            self.global_mean
            + self.user_biases[u]
            + self.item_biases[i]
            + np.dot(self.user_factors[u], self.item_factors[i])
        )
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, user_id, product_id):
        if user_id not in self.user_to_idx or product_id not in self.item_to_idx:
            return self.global_mean
        return self._predict_idx(self.user_to_idx[user_id], self.item_to_idx[product_id])

    def recommend(self, user_id, top_n=10):
        if user_id not in self.user_to_idx:
            return []

        u = self.user_to_idx[user_id]
        rated = set(self.ratings_df[self.ratings_df["user_id"] == user_id]["product_id"])

        preds = []
        for item_id in self.items:
            if item_id in rated:
                continue
            i = self.item_to_idx[item_id]
            score = self._predict_idx(u, i)
            preds.append((item_id, score))

        preds.sort(key=lambda x: x[1], reverse=True)

        results = []
        for pid, score in preds[:top_n]:
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
                "reason": f"Predicted rating: {score:.1f}/5 from latent preference model",
            })
        return results
