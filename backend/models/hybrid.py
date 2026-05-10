"""
MindMatch – Hybrid Recommender

Primary:  MindMatchHybrid
          score = α × CBF_score + (1−α) × CF_score
          Default α = 0.5 (equal weight, as described in the project proposal)

Legacy:   HybridRecommender (3-model blend, kept for comparison tabs)
"""

import numpy as np


class MindMatchHybrid:
    """
    Two-model hybrid as described in the MindMatch proposal:
        final_score = alpha × content_based + (1 - alpha) × collaborative_svd
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha     = alpha          # weight for Content-Based component
        self.cbf_model = None
        self.cf_model  = None           # Surprise SVD recommender
        self._products = None

    def set_models(self, cbf_model, cf_model, products_df=None):
        self.cbf_model = cbf_model
        self.cf_model  = cf_model
        self._products = products_df

    @staticmethod
    def _normalise(recs: list) -> dict:
        """Min-max normalise scores to [0, 1]."""
        if not recs:
            return {}
        scores = {r["product_id"]: float(r["score"]) for r in recs}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def predict(self, user_id: str, product_id: str) -> float:
        cbf = (self.cbf_model.predict(user_id, product_id)
               if self.cbf_model and hasattr(self.cbf_model, "predict") else 0.5)
        cf  = (self.cf_model.predict(user_id, product_id)
               if self.cf_model  and hasattr(self.cf_model,  "predict") else 0.5)
        return self.alpha * cbf + (1 - self.alpha) * cf

    def recommend(self, user_id: str, top_n: int = 10) -> list:
        cbf_recs = self.cbf_model.recommend(user_id, top_n=60) if self.cbf_model else []
        cf_recs  = self.cf_model.recommend(user_id,  top_n=60) if self.cf_model  else []

        cbf_scores = self._normalise(cbf_recs)
        cf_scores  = self._normalise(cf_recs)

        all_pids = set(cbf_scores) | set(cf_scores)

        blended = {}
        for pid in all_pids:
            blended[pid] = (
                self.alpha * cbf_scores.get(pid, 0.0)
                + (1 - self.alpha) * cf_scores.get(pid, 0.0)
            )

        top = sorted(blended.items(), key=lambda x: x[1], reverse=True)[:top_n]

        # Build a product-info lookup from whichever model returned results
        info_lookup = {}
        for rec in cbf_recs + cf_recs:
            pid = rec["product_id"]
            if pid not in info_lookup:
                info_lookup[pid] = rec

        results = []
        for pid, score in top:
            if pid not in info_lookup:
                continue
            item           = info_lookup[pid].copy()
            item["score"]  = round(float(score), 4)
            item["reason"] = (
                f"Hybrid: {self.alpha:.0%} Content-Based + "
                f"{1 - self.alpha:.0%} SVD Collaborative"
            )
            results.append(item)
        return results


# ---------------------------------------------------------------------------
# Legacy 3-model hybrid (kept for comparison in the evaluation tab)
# ---------------------------------------------------------------------------

class HybridRecommender:
    """
    Weighted blend of User-CF, Matrix Factorization, and Content-Based scores.
    Kept for backward compatibility / evaluation comparison.
    """

    def __init__(self, cf_weight=0.35, mf_weight=0.40, cb_weight=0.25):
        self.cf_weight = cf_weight
        self.mf_weight = mf_weight
        self.cb_weight = cb_weight
        self.cf_model  = None
        self.mf_model  = None
        self.cb_model  = None

    def set_models(self, cf_model, mf_model, cb_model):
        self.cf_model = cf_model
        self.mf_model = mf_model
        self.cb_model = cb_model

    @staticmethod
    def _normalise(recs):
        if not recs:
            return {}
        scores = {r["product_id"]: r["score"] for r in recs}
        lo, hi = min(scores.values()), max(scores.values())
        if hi == lo:
            return {k: 1.0 for k in scores}
        return {k: (v - lo) / (hi - lo) for k, v in scores.items()}

    def recommend(self, user_id, top_n=10):
        cf_recs = self.cf_model.recommend(user_id, top_n=60) if self.cf_model else []
        mf_recs = self.mf_model.recommend(user_id, top_n=60) if self.mf_model else []
        cb_recs = self.cb_model.recommend(user_id, top_n=60) if self.cb_model else []

        cf_s = self._normalise(cf_recs)
        mf_s = self._normalise(mf_recs)
        cb_s = self._normalise(cb_recs)

        all_pids = set(cf_s) | set(mf_s) | set(cb_s)

        hybrid = {
            pid: (
                self.cf_weight * cf_s.get(pid, 0.0)
                + self.mf_weight * mf_s.get(pid, 0.0)
                + self.cb_weight * cb_s.get(pid, 0.0)
            )
            for pid in all_pids
        }

        top = sorted(hybrid.items(), key=lambda x: x[1], reverse=True)[:top_n]

        info_lookup = {}
        for rec in cf_recs + mf_recs + cb_recs:
            pid = rec["product_id"]
            if pid not in info_lookup:
                info_lookup[pid] = rec

        results = []
        for pid, score in top:
            if pid not in info_lookup:
                continue
            item           = info_lookup[pid].copy()
            item["score"]  = round(float(score), 3)
            item["reason"] = "Recommended by hybrid analysis (CF 35% + MF 40% + Content 25%)"
            results.append(item)
        return results
