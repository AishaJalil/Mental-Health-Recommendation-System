import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

import numpy as np
import pandas as pd
from collections import defaultdict
from flask import Flask, jsonify, request
from flask_cors import CORS

from data.generate_data import generate_data
from nlp.condition_parser import parse_condition
from models.collaborative_filtering import CollaborativeFilteringRecommender
from models.content_based import ContentBasedRecommender
from models.matrix_factorization import MatrixFactorizationRecommender
from models.cf_surprise import SurpriseSVDRecommender
from models.hybrid import MindMatchHybrid, HybridRecommender
from models.neural_cf import NeuralCFRecommender
from evaluation.metrics import evaluate_ranking, evaluate_rating_prediction
from evaluation.fairness import evaluate_fairness

app = Flask(__name__)
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

products_df = users_df = ratings_df = None

# MindMatch primary models
svd_cf  = None   # Surprise SVD — main CF model
cb      = None   # Content-Based (TF-IDF + cosine)
mm_hybrid = None  # MindMatch Hybrid: α × CBF + (1−α) × CF

# Supplementary models
cf_user = cf_item = mf = legacy_hybrid = ncf = None

_eval_cache     = {}   # cleared on restart; recomputed on first /api/evaluation request
_fairness_cache = {}


# ---------------------------------------------------------------------------
# Startup: load / generate data, then train all models
# ---------------------------------------------------------------------------

def load_data():
    global products_df, users_df, ratings_df
    p = os.path.join(DATA_DIR, "products.csv")
    if not os.path.exists(p):
        print("Data not found – generating MindMatch dataset …")
        products_df, users_df, ratings_df = generate_data(n_users=1000)
    else:
        products_df = pd.read_csv(os.path.join(DATA_DIR, "products.csv"))
        users_df    = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
        ratings_df  = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    print(f"Data loaded: {len(products_df)} activities | {len(users_df)} users | {len(ratings_df)} interactions")


def train_models():
    global svd_cf, cb, mm_hybrid, cf_user, cf_item, mf, legacy_hybrid, ncf

    # ── MindMatch primary models ────────────────────────────────────────────
    print("Training Content-Based Filtering (TF-IDF + cosine) …")
    cb = ContentBasedRecommender()
    cb.fit(products_df, ratings_df)

    print("Training SVD Collaborative Filtering (Surprise) …")
    svd_cf = SurpriseSVDRecommender(n_factors=100, n_epochs=20)
    svd_cf.fit(ratings_df, products_df)

    print("Building MindMatch Hybrid (α=0.5: 50% CBF + 50% SVD-CF) …")
    mm_hybrid = MindMatchHybrid(alpha=0.5)
    mm_hybrid.set_models(cbf_model=cb, cf_model=svd_cf, products_df=products_df)

    # ── Supplementary models ────────────────────────────────────────────────
    print("Training User-Based Collaborative Filtering …")
    cf_user = CollaborativeFilteringRecommender(method="user")
    cf_user.fit(ratings_df, products_df)

    print("Training Item-Based Collaborative Filtering …")
    cf_item = CollaborativeFilteringRecommender(method="item")
    cf_item.fit(ratings_df, products_df)

    print("Training Matrix Factorization (SGD / fallback) …")
    mf = MatrixFactorizationRecommender(n_factors=50, n_epochs=20, lr=0.005, reg=0.02)
    mf.fit(ratings_df, products_df)

    print("Building Legacy Hybrid (CF 35% + MF 40% + CBF 25%) …")
    legacy_hybrid = HybridRecommender(cf_weight=0.35, mf_weight=0.40, cb_weight=0.25)
    legacy_hybrid.set_models(cf_user, mf, cb)

    print("Training Neural Collaborative Filtering (NeuMF) …")
    ncf = NeuralCFRecommender(
        gmf_factors=16, mlp_factors=16, mlp_hidden=(32, 16),
        lr=0.005, n_epochs=15, batch_size=512, neg_ratio=4, reg=0.001,
    )
    ncf.fit(ratings_df, products_df)

    print("All models ready.\n")


print("=" * 50)
print("  MindMatch – Stress Activity Recommender")
print("=" * 50)
load_data()
train_models()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _popularity_scores():
    stats = ratings_df.groupby("product_id").agg(
        avg_rating=("rating", "mean"),
        n_ratings=("rating", "count"),
    ).reset_index()
    C = stats["avg_rating"].mean()
    m = stats["n_ratings"].quantile(0.25)
    stats["score"] = (stats["n_ratings"] * stats["avg_rating"] + m * C) / (stats["n_ratings"] + m)
    return stats


def _enrich_recs(recs):
    """Attach wellness-specific fields to recommendation dicts."""
    extra_cols = ["duration_minutes", "stress_target", "effectiveness"]
    available  = [c for c in extra_cols if c in products_df.columns]
    if not available:
        return recs
    lookup = products_df.set_index("product_id")[available].to_dict("index")
    for r in recs:
        info = lookup.get(r["product_id"], {})
        for col in available:
            r[col] = info.get(col)
        if not r.get("duration_minutes"):
            r["duration_minutes"] = int(r.get("price", 0))
    return recs


# ---------------------------------------------------------------------------
# General routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "project": "MindMatch"})


@app.route("/api/users")
def get_users():
    return jsonify(users_df.to_dict(orient="records"))


@app.route("/api/products")
def get_products():
    cat = request.args.get("category")
    df  = products_df if not cat else products_df[products_df["category"] == cat]
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/categories")
def get_categories():
    return jsonify(products_df["category"].unique().tolist())


@app.route("/api/stats")
def get_stats():
    return jsonify({
        "total_users":      int(len(users_df)),
        "total_products":   int(len(products_df)),
        "total_activities": int(len(products_df)),
        "total_ratings":    int(len(ratings_df)),
        "avg_rating":       round(float(ratings_df["rating"].mean()), 2),
        "sparsity":         round(
            (1 - len(ratings_df) / (len(users_df) * len(products_df))) * 100, 1
        ),
        "categories": {
            k: int(v) for k, v in products_df["category"].value_counts().items()
        },
        "rating_distribution": {
            str(k): int(v)
            for k, v in ratings_df["rating"].value_counts().sort_index().items()
        },
    })


@app.route("/api/user/<user_id>/history")
def user_history(user_id):
    ur   = ratings_df[ratings_df["user_id"] == user_id].copy()
    ur   = ur.merge(products_df, on="product_id").sort_values("rating", ascending=False)
    cols = ["product_id", "name", "category", "price", "rating", "timestamp"]
    return jsonify(ur[cols].to_dict(orient="records"))


# ---------------------------------------------------------------------------
# Recommendation routes — MindMatch primary models
# ---------------------------------------------------------------------------

@app.route("/api/recommend/hybrid/<user_id>")
def rec_hybrid(user_id):
    n = int(request.args.get("top_n", 10))
    return jsonify({
        "user_id":    user_id,
        "algorithm":  "MindMatch Hybrid (CBF + SVD)",
        "description": "score = 0.5 × Content-Based + 0.5 × SVD Collaborative Filtering",
        "recommendations": _enrich_recs(mm_hybrid.recommend(user_id, top_n=n)),
    })


@app.route("/api/recommend/content-based/<user_id>")
def rec_cb(user_id):
    n = int(request.args.get("top_n", 10))
    return jsonify({
        "user_id":    user_id,
        "algorithm":  "Content-Based Filtering",
        "description": "TF-IDF profile from your rated activities matched via cosine similarity.",
        "recommendations": _enrich_recs(cb.recommend(user_id, top_n=n)),
    })


@app.route("/api/recommend/svd-cf/<user_id>")
def rec_svd_cf(user_id):
    n = int(request.args.get("top_n", 10))
    return jsonify({
        "user_id":    user_id,
        "algorithm":  "SVD Collaborative Filtering (Surprise)",
        "description": "Latent-factor SVD from the Surprise library — predicts rating for each unseen activity.",
        "recommendations": _enrich_recs(svd_cf.recommend(user_id, top_n=n)),
    })


# ---------------------------------------------------------------------------
# Recommendation routes — supplementary models
# ---------------------------------------------------------------------------

@app.route("/api/recommend/user-cf/<user_id>")
def rec_user_cf(user_id):
    n = int(request.args.get("top_n", 10))
    return jsonify({
        "user_id":    user_id,
        "algorithm":  "User-Based Collaborative Filtering",
        "description": "Finds users with similar stress profiles and recommends activities they found helpful.",
        "recommendations": _enrich_recs(cf_user.recommend(user_id, top_n=n)),
    })


@app.route("/api/recommend/item-cf/<user_id>")
def rec_item_cf(user_id):
    n = int(request.args.get("top_n", 10))
    return jsonify({
        "user_id":    user_id,
        "algorithm":  "Item-Based Collaborative Filtering",
        "description": "Recommends activities similar to ones you have already found helpful.",
        "recommendations": _enrich_recs(cf_item.recommend(user_id, top_n=n)),
    })


@app.route("/api/recommend/matrix-factorization/<user_id>")
def rec_mf(user_id):
    n = int(request.args.get("top_n", 10))
    return jsonify({
        "user_id":    user_id,
        "algorithm":  "Matrix Factorization (SGD)",
        "description": "Custom SGD-trained latent-factor model for rating prediction.",
        "recommendations": _enrich_recs(mf.recommend(user_id, top_n=n)),
    })


@app.route("/api/recommend/neural-cf/<user_id>")
def rec_neural_cf(user_id):
    n = int(request.args.get("top_n", 10))
    return jsonify({
        "user_id":    user_id,
        "algorithm":  "Neural Collaborative Filtering (NeuMF)",
        "description": "NeuMF: combines Generalized Matrix Factorization (GMF) with an MLP on user/item embeddings, trained on implicit feedback via negative sampling.",
        "recommendations": _enrich_recs(ncf.recommend(user_id, top_n=n)),
    })


@app.route("/api/recommend/popular")
def rec_popular():
    n     = int(request.args.get("top_n", 10))
    stats = _popularity_scores()
    top   = stats.sort_values("score", ascending=False).head(n)
    top   = top.merge(products_df, on="product_id")
    results = []
    for _, row in top.iterrows():
        results.append({
            "product_id":       row["product_id"],
            "name":             row["name"],
            "category":         row["category"],
            "price":            float(row["price"]),
            "duration_minutes": int(row["price"]),
            "stress_target":    row.get("stress_target", "all"),
            "effectiveness":    row.get("effectiveness", "moderate"),
            "score":            round(float(row["score"]), 3),
            "avg_rating":       round(float(row["avg_rating"]), 2),
            "n_ratings":        int(row["n_ratings"]),
            "reason":           f"Rated by {int(row['n_ratings'])} users – avg {row['avg_rating']:.1f}/5",
        })
    return jsonify({"algorithm": "Popularity-Based (Bayesian Average)", "recommendations": results})


@app.route("/api/similar/<product_id>")
def similar(product_id):
    n = int(request.args.get("top_n", 6))
    return jsonify({
        "product_id":    product_id,
        "content_based": cb.get_similar_products(product_id, top_n=n),
        "collaborative": svd_cf.get_similar_products(product_id, top_n=n)
                         or cf_item.get_similar_products(product_id, top_n=n),
    })


@app.route("/api/compare-all/<user_id>")
def compare_all(user_id):
    top_n = int(request.args.get("top_n", 10))

    model_entries = [
        ("Hybrid",        mm_hybrid),
        ("SVD-CF",        svd_cf),
        ("Content-Based", cb),
        ("User-CF",       cf_user),
        ("Item-CF",       cf_item),
        ("Neural CF",     ncf),
    ]

    models_recs = {}
    for label, model in model_entries:
        try:
            models_recs[label] = _enrich_recs(model.recommend(user_id, top_n=top_n))
        except Exception:
            models_recs[label] = []

    # Aggregate: item → which models recommended it + their scores
    item_models  = defaultdict(list)
    item_scores  = defaultdict(list)
    item_info    = {}

    for label, recs in models_recs.items():
        for rec in recs:
            pid = rec["product_id"]
            item_models[pid].append(label)
            item_scores[pid].append(rec["score"])
            if pid not in item_info:
                item_info[pid] = {k: v for k, v in rec.items()
                                  if k not in ("score", "reason")}

    consensus = []
    for pid, model_list in item_models.items():
        avg_score = float(np.mean(item_scores[pid]))
        entry = item_info[pid].copy()
        entry["n_models"] = len(model_list)
        entry["avg_score"] = round(avg_score, 4)
        entry["models"]   = model_list
        consensus.append(entry)

    consensus.sort(key=lambda x: (x["n_models"], x["avg_score"]), reverse=True)

    # Pairwise agreement: # items in common between every pair of models
    all_labels = [label for label, _ in model_entries]
    pid_sets   = {label: {r["product_id"] for r in recs}
                  for label, recs in models_recs.items()}
    agreement  = {
        la: {
            lb: (top_n if la == lb else len(pid_sets[la] & pid_sets[lb]))
            for lb in all_labels
        }
        for la in all_labels
    }

    return jsonify({
        "user_id":   user_id,
        "top_n":     top_n,
        "models":    models_recs,
        "consensus": consensus,
        "agreement": agreement,
    })


# ---------------------------------------------------------------------------
# Evaluation route — Precision@5, Recall@5, NDCG@5, Coverage, Diversity
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Condition-based recommendation (new / anonymous users — cold start)
# ---------------------------------------------------------------------------

def _classify_severity(score, dimension):
    thresholds = {
        "depression": [(10, "Mild"), (14, "Moderate"), (21, "Severe"), (28, "Extremely Severe")],
        "anxiety":    [( 8, "Mild"), (10, "Moderate"), (15, "Severe"), (20, "Extremely Severe")],
        "stress":     [(15, "Mild"), (19, "Moderate"), (26, "Severe"), (34, "Extremely Severe")],
    }
    label = "Normal"
    for threshold, lbl in thresholds[dimension]:
        if score >= threshold:
            label = lbl
    return label


def _find_nearest_user(depression, anxiety, stress):
    """
    Find the existing user whose DASS profile is closest to the given scores.
    Uses Euclidean distance on (dass_depression, dass_anxiety, dass_stress).
    """
    target = np.array([depression, anxiety, stress], dtype=float)
    scores = users_df[["dass_depression", "dass_anxiety", "dass_stress"]].values.astype(float)
    dists  = np.sqrt(((scores - target) ** 2).sum(axis=1))
    idx    = int(np.argmin(dists))
    return users_df.iloc[idx]["user_id"], round(float(dists[idx]), 2)


def _condition_recommend(depression, anxiety, stress, top_n=10):
    eff_mult = {"high": 1.3, "moderate": 1.0, "low": 0.7}
    dep_n = depression / 42.0
    anx_n = anxiety   / 42.0
    str_n = stress    / 42.0

    pop = ratings_df.groupby("product_id")["rating"].mean().to_dict()

    sev_dep = _classify_severity(depression, "depression")
    sev_anx = _classify_severity(anxiety,    "anxiety")
    sev_str = _classify_severity(stress,     "stress")
    sev_map = {"depression": sev_dep, "anxiety": sev_anx, "stress": sev_str}

    results = []
    for _, row in products_df.iterrows():
        target = str(row.get("stress_target", "all")).lower()
        eff    = str(row.get("effectiveness", "moderate")).lower()

        if   target == "depression": base = dep_n
        elif target == "anxiety":    base = anx_n
        elif target == "stress":     base = str_n
        else:                        base = (dep_n + anx_n + str_n) / 3.0

        mult    = eff_mult.get(eff, 1.0)
        pop_avg = pop.get(row["product_id"], 3.0)
        score   = base * mult + (pop_avg / 5.0) * 0.15

        if target == "all":
            reason = f"Supports all conditions · {eff} effectiveness"
        else:
            sev = sev_map.get(target, "")
            reason = f"Targets {target} ({sev.lower()}) · {eff} effectiveness"

        results.append({
            "product_id": row["product_id"],
            "name":       row["name"],
            "category":   row["category"],
            "price":      float(row["price"]),
            "score":      round(float(score), 4),
            "reason":     reason,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


_MODEL_MAP = {
    "content-based": lambda: cb,
    "hybrid":        lambda: mm_hybrid,
    "svd-cf":        lambda: svd_cf,
    "user-cf":       lambda: cf_user,
    "neural-cf":     lambda: ncf,
}

@app.route("/api/recommend/by-condition", methods=["POST"])
def rec_by_condition():
    data       = request.json or {}
    depression = max(0, min(42, int(data.get("depression", 0))))
    anxiety    = max(0, min(42, int(data.get("anxiety",    0))))
    stress     = max(0, min(42, int(data.get("stress",     0))))
    top_n      = min(20, max(1,  int(data.get("top_n",    10))))
    model_key  = data.get("model", "condition-based")

    severity = {
        "depression": _classify_severity(depression, "depression"),
        "anxiety":    _classify_severity(anxiety,    "anxiety"),
        "stress":     _classify_severity(stress,     "stress"),
    }

    # ── Condition-based (rule-based, no user_id needed) ──────────────────────
    if model_key == "condition-based" or model_key not in _MODEL_MAP:
        return jsonify({
            "algorithm":    "Condition-Based Recommender",
            "description":  "Activities scored by DASS dimension match × clinical effectiveness.",
            "model_key":    "condition-based",
            "nearest_user": None,
            "severity":     severity,
            "recommendations": _enrich_recs(_condition_recommend(depression, anxiety, stress, top_n)),
        })

    # ── ML model via nearest-neighbour cold-start ─────────────────────────────
    nearest_uid, distance = _find_nearest_user(depression, anxiety, stress)
    nearest_row  = users_df[users_df["user_id"] == nearest_uid].iloc[0]
    nearest_info = {
        "user_id":    nearest_uid,
        "name":       nearest_row.get("name", nearest_uid),
        "age_group":  nearest_row.get("age_group", ""),
        "occupation": nearest_row.get("occupation", ""),
        "dass_depression": int(nearest_row.get("dass_depression", 0)),
        "dass_anxiety":    int(nearest_row.get("dass_anxiety",    0)),
        "dass_stress":     int(nearest_row.get("dass_stress",     0)),
        "distance":   distance,
    }

    model = _MODEL_MAP[model_key]()
    recs  = model.recommend(nearest_uid, top_n=top_n)

    algo_labels = {
        "content-based": "Content-Based (SBERT)",
        "hybrid":        "MindMatch Hybrid",
        "svd-cf":        "SVD Collaborative Filtering",
        "user-cf":       "User-Based CF",
        "neural-cf":     "Neural CF (NeuMF)",
    }

    return jsonify({
        "algorithm":    algo_labels.get(model_key, model_key),
        "description":  f"Recommendations for the most similar user in the dataset (DASS distance: {distance}).",
        "model_key":    model_key,
        "nearest_user": nearest_info,
        "severity":     severity,
        "recommendations": _enrich_recs(recs),
    })


# ---------------------------------------------------------------------------
# Evaluation route — Precision@5, Recall@5, NDCG@5, Coverage, Diversity
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# NLP route — parse free-text condition description into DASS scores
# ---------------------------------------------------------------------------

@app.route("/api/nlp/parse-condition", methods=["POST"])
def nlp_parse_condition():
    data = request.json or {}
    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "text is required"}), 400
    if len(text) > 2000:
        return jsonify({"error": "text too long (max 2000 chars)"}), 400

    result = parse_condition(text)
    result["severity"] = {
        "depression": _classify_severity(result["depression"], "depression"),
        "anxiety":    _classify_severity(result["anxiety"],    "anxiety"),
        "stress":     _classify_severity(result["stress"],     "stress"),
    }
    return jsonify(result)


@app.route("/api/evaluation")
def evaluation():
    global _eval_cache
    if _eval_cache:
        return jsonify(_eval_cache)

    print("Running offline evaluation (K=5) …")
    result = {}

    for label, model in [
        ("MindMatch Hybrid",     mm_hybrid),
        ("SVD-CF (Surprise)",    svd_cf),
        ("Content-Based",        cb),
        ("User-Based CF",        cf_user),
        ("Item-Based CF",        cf_item),
        ("Neural CF",            ncf),
    ]:
        metrics = evaluate_ranking(model, ratings_df, products_df=products_df, k=5, n_users=60)
        if label in ("SVD-CF (Surprise)", "MindMatch Hybrid"):
            rating_m = evaluate_rating_prediction(svd_cf, ratings_df, n_users=60)
            metrics.update(rating_m)
        result[label] = metrics

    _eval_cache = result
    return jsonify(result)


# ---------------------------------------------------------------------------
# Fairness route — Demographic Parity across age_group and occupation
# ---------------------------------------------------------------------------

@app.route("/api/fairness")
def fairness():
    global _fairness_cache
    if _fairness_cache:
        return jsonify(_fairness_cache)

    print("Running fairness evaluation (K=5, n=200 users) …")
    result = {}
    for label, model in [
        ("MindMatch Hybrid", mm_hybrid),
        ("SVD-CF",           svd_cf),
        ("Content-Based",    cb),
    ]:
        result[label] = evaluate_fairness(
            model, ratings_df, users_df, products_df, k=5, n_users=200
        )

    _fairness_cache = result
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
