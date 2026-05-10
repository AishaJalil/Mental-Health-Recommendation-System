"""
MindMatch – Fairness Evaluation (Demographic Parity)

Measures whether NDCG@k is equal across demographic groups (age_group, occupation).
A fair model serves all groups equally well.

Key metrics:
  - NDCG Gap           : max_group_NDCG − min_group_NDCG  (lower = fairer, ideal = 0)
  - Disparate Impact   : min_group_NDCG / max_group_NDCG  (higher = fairer, threshold = 0.8)
"""

import numpy as np
import pandas as pd


def _ndcg_at_k(recommended_ids: list, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    dcg  = sum(1.0 / np.log2(i + 2) for i, pid in enumerate(recommended_ids[:k]) if pid in relevant_ids)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _group_stats(scores: list[float]) -> dict:
    lo, hi = min(scores), max(scores)
    return {
        "ndcg_gap":         round(hi - lo, 4),
        "disparate_impact": round(lo / hi, 4) if hi > 0 else 1.0,
        "is_fair":          (lo / hi >= 0.8) if hi > 0 else True,
    }


def evaluate_fairness(model, ratings_df, users_df, products_df, k=5, n_users=200) -> dict:
    """
    For each sampled user compute NDCG@k, then aggregate by demographic group.
    Returns per-group NDCG scores + gap / disparate-impact fairness metrics.
    """
    sample  = users_df.sample(min(n_users, len(users_df)), random_state=42)
    records = []

    for _, user in sample.iterrows():
        uid        = user["user_id"]
        age_group  = str(user.get("age_group",  "Unknown"))
        occupation = str(user.get("occupation", "Unknown"))

        ur = ratings_df[ratings_df["user_id"] == uid]
        if len(ur) < 5:
            continue

        # Hold-out: top-rated items as ground-truth relevant set
        n_test   = max(1, len(ur) // 5)
        relevant = set(ur.nlargest(n_test, "rating")["product_id"].tolist())

        try:
            recs    = model.recommend(uid, top_n=k)
            rec_ids = [r["product_id"] for r in recs]
        except Exception:
            continue

        records.append({
            "age_group":  age_group,
            "occupation": occupation,
            "ndcg":       _ndcg_at_k(rec_ids, relevant, k),
        })

    if not records:
        return {}

    df = pd.DataFrame(records)

    def build_group_result(col: str) -> dict:
        grp_mean = df.groupby(col)["ndcg"].mean().round(4)
        stats    = _group_stats(grp_mean.tolist())
        return {
            "scores": {k: round(float(v), 4) for k, v in grp_mean.items()},
            **stats,
        }

    return {
        "age_group":         build_group_result("age_group"),
        "occupation":        build_group_result("occupation"),
        "overall_ndcg":      round(float(df["ndcg"].mean()), 4),
        "n_users_evaluated": len(records),
    }
