"""
MindMatch – Evaluation Metrics
Metrics: Precision@5, Recall@5, NDCG@5, Coverage, Diversity, RMSE, MAE
"""

import numpy as np
from sklearn.model_selection import train_test_split


def rmse(actual, predicted):
    a, p = np.array(actual), np.array(predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual, predicted):
    a, p = np.array(actual), np.array(predicted)
    return float(np.mean(np.abs(a - p)))


def precision_at_k(recommended, relevant, k):
    top = recommended[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return len(set(recommended[:k]) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    top  = recommended[:k]
    dcg  = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top) if item in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def coverage(all_rec_lists, catalog_size):
    """Fraction of catalogue items that appear in any recommendation list."""
    recommended_items = set()
    for rec_list in all_rec_lists:
        recommended_items.update(rec_list)
    return round(len(recommended_items) / catalog_size, 4) if catalog_size else 0.0


def intra_list_diversity(rec_list, products_df):
    """
    Average pairwise category-dissimilarity within a recommendation list.
    Two items in the same category count as 0 (similar); different = 1.
    Range: [0, 1] — higher means more diverse recommendations.
    """
    if len(rec_list) < 2:
        return 0.0

    cat_map = products_df.set_index("product_id")["category"].to_dict()
    cats    = [cat_map.get(pid, "") for pid in rec_list]

    total = pairs = 0
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            total += 1
            if cats[i] != cats[j]:
                pairs += 1
    return round(pairs / total, 4) if total else 0.0


def evaluate_ranking(model, ratings_df, products_df=None, k=5, n_users=60):
    """
    Compute Precision@K, Recall@K, NDCG@K, Coverage, Diversity via train/test split.

    Uses model.predict() to score ALL catalog items so that test-relevant items
    are never excluded by the model's seen-item filter.  Falls back to
    model.recommend() for models that lack predict().
    """
    _, test = train_test_split(ratings_df, test_size=0.2, random_state=42)

    all_products = ratings_df["product_id"].unique().tolist()
    catalog_size = len(all_products)
    has_predict  = hasattr(model, "predict")

    p_list, r_list, n_list, div_list = [], [], [], []
    all_rec_lists = []

    for user_id in test["user_id"].unique()[:n_users]:
        user_test = test[test["user_id"] == user_id]
        relevant  = user_test[user_test["rating"] >= 4.0]["product_id"].tolist()
        if not relevant:
            continue
        try:
            if has_predict:
                # Score every item without the seen-item filter
                scores  = {pid: model.predict(user_id, pid) for pid in all_products}
                rec_ids = sorted(scores, key=lambda p: scores[p], reverse=True)[:k]
            else:
                recs    = model.recommend(user_id, top_n=k)
                rec_ids = [r["product_id"] for r in recs]

            if not rec_ids:
                continue
            p_list.append(precision_at_k(rec_ids, relevant, k))
            r_list.append(recall_at_k(rec_ids, relevant, k))
            n_list.append(ndcg_at_k(rec_ids, relevant, k))
            all_rec_lists.append(rec_ids)
            if products_df is not None:
                div_list.append(intra_list_diversity(rec_ids, products_df))
        except Exception:
            continue

    cov = coverage(all_rec_lists, catalog_size)
    div = round(float(np.mean(div_list)), 4) if div_list else 0.0

    return {
        "precision_at_k":      round(float(np.mean(p_list)), 4) if p_list else 0.0,
        "recall_at_k":         round(float(np.mean(r_list)), 4) if r_list else 0.0,
        "ndcg_at_k":           round(float(np.mean(n_list)), 4) if n_list else 0.0,
        "coverage":            cov,
        "diversity":           div,
        "k":                   k,
        "n_users_evaluated":   len(p_list),
    }


def evaluate_rating_prediction(model, ratings_df, n_users=60):
    """Compute RMSE and MAE for rating prediction (SVD CF / MF models)."""
    _, test = train_test_split(ratings_df, test_size=0.2, random_state=42)

    actuals, preds = [], []
    for user_id in test["user_id"].unique()[:n_users]:
        for _, row in test[test["user_id"] == user_id].iterrows():
            try:
                pred = model.predict(row["user_id"], row["product_id"])
                actuals.append(float(row["rating"]))
                preds.append(float(pred))
            except Exception:
                continue

    if not actuals:
        return {"rmse": None, "mae": None}
    return {
        "rmse": round(rmse(actuals, preds), 4),
        "mae":  round(mae(actuals, preds),  4),
    }
