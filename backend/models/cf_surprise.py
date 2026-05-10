"""
MindMatch – Collaborative Filtering via Surprise library SVD.
Primary CF model as specified in the project description.

Falls back to a mean-based predictor if scikit-surprise is not installed.
Install: pip install scikit-surprise
"""

import numpy as np
import pandas as pd

try:
    from surprise import SVD, Dataset, Reader
    _SURPRISE_OK = True
except ImportError:
    _SURPRISE_OK = False


class SurpriseSVDRecommender:
    """
    Wraps Surprise SVD for the MindMatch CF component.
    Interface is identical to the other recommenders so app.py can swap freely.
    """

    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr_all    = lr_all
        self.reg_all   = reg_all

        self._model       = None
        self._trainset    = None
        self._products    = None
        self._user_means  = {}
        self._global_mean = 3.0

    def fit(self, ratings_df: pd.DataFrame, products_df: pd.DataFrame):
        self._products    = products_df.copy()
        self._global_mean = float(ratings_df["rating"].mean())
        for uid, grp in ratings_df.groupby("user_id"):
            self._user_means[uid] = float(grp["rating"].mean())

        if not _SURPRISE_OK:
            print("  [WARNING] scikit-surprise not installed – SVD CF uses mean-fallback.")
            print("            Fix: pip install scikit-surprise  then restart the server.")
            return

        reader         = Reader(rating_scale=(1.0, 5.0))
        data           = Dataset.load_from_df(
            ratings_df[["user_id", "product_id", "rating"]], reader
        )
        self._trainset = data.build_full_trainset()
        self._model    = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            verbose=False,
        )
        self._model.fit(self._trainset)

    # ------------------------------------------------------------------
    def _predict_raw(self, user_id: str, product_id: str) -> float:
        """Returns predicted rating (1–5 scale)."""
        if self._model is None:
            return self._user_means.get(user_id, self._global_mean)
        try:
            return float(self._model.predict(user_id, product_id).est)
        except Exception:
            return self._user_means.get(user_id, self._global_mean)

    def predict(self, user_id: str, product_id: str) -> float:
        """Alias used by the evaluation module."""
        return self._predict_raw(user_id, product_id)

    def recommend(self, user_id: str, top_n: int = 10) -> list:
        if self._products is None:
            return []

        preds = [
            (row, self._predict_raw(user_id, row["product_id"]))
            for _, row in self._products.iterrows()
        ]
        preds.sort(key=lambda x: x[1], reverse=True)

        results = []
        for row, raw_score in preds[:top_n]:
            results.append({
                "product_id": row["product_id"],
                "name":       row["name"],
                "category":   row["category"],
                "price":      float(row.get("price", row.get("duration_minutes", 0))),
                "score":      round(raw_score / 5.0, 4),   # normalise to 0-1
                "reason":     f"SVD predicted rating: {raw_score:.2f}/5",
            })
        return results

    def get_similar_products(self, product_id: str, top_n: int = 6) -> list:
        """Cosine similarity in latent item-factor space."""
        if self._model is None or self._trainset is None:
            return []
        try:
            inner = self._trainset.to_inner_iid(product_id)
            qi    = self._model.qi[inner]
            sims  = []
            for other_inner in range(len(self._model.qi)):
                if other_inner == inner:
                    continue
                qi_other = self._model.qi[other_inner]
                denom    = np.linalg.norm(qi) * np.linalg.norm(qi_other)
                if denom == 0:
                    continue
                sim     = float(np.dot(qi, qi_other) / denom)
                raw_pid = self._trainset.to_raw_iid(other_inner)
                sims.append((raw_pid, sim))
            sims.sort(key=lambda x: x[1], reverse=True)

            results = []
            for pid, sim in sims[:top_n]:
                row = self._products[self._products["product_id"] == pid]
                if row.empty:
                    continue
                r = row.iloc[0]
                results.append({
                    "product_id": pid,
                    "name":       r["name"],
                    "category":   r.get("category", ""),
                    "similarity": round(sim, 4),
                })
            return results
        except Exception:
            return []
