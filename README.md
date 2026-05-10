# MindMatch – Mental Health Recommender System

A full-stack recommendation system that suggests personalised stress-management activities to users based on their DASS-21 (Depression, Anxiety, Stress Scales) scores. The system implements and compares **six distinct recommendation algorithms**, includes an NLP-based condition parser, offline evaluation metrics, and a fairness audit dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Recommendation Algorithms](#recommendation-algorithms)
- [Dataset](#dataset)
- [Evaluation Metrics](#evaluation-metrics)
- [Fairness Analysis](#fairness-analysis)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [API Reference](#api-reference)

---

## Overview

MindMatch addresses the growing need for personalised mental wellness support. Rather than generic advice, it tailors activity recommendations to each user's specific psychological profile as measured by the clinically validated DASS-21 instrument.

Users can receive recommendations either by:
1. Selecting an existing user profile (collaborative/content-based filtering), or
2. Entering their own DASS-21 scores manually via sliders, or
3. Describing their mental state in plain language (NLP parsing via Google Gemini AI with keyword fallback).

The primary model is a **hybrid recommender** that blends Content-Based Filtering and SVD Collaborative Filtering in equal weights (α = 0.5), while five additional algorithms are provided for comparison and evaluation.

---

## Features

| Feature | Description |
|---|---|
| 6 Recommendation Algorithms | Hybrid, SVD-CF, Content-Based, User-CF, Item-CF, Neural CF |
| NLP Condition Parser | Free-text input → DASS-21 scores via Gemini AI or keyword matching |
| Manual DASS Sliders | Direct score input (0–42) with real-time severity classification |
| Offline Evaluation Dashboard | Precision@5, Recall@5, NDCG@5, Coverage, Diversity, RMSE, MAE |
| Fairness Audit | Demographic parity across age groups and occupations (4/5ths rule) |
| Model Comparison View | Consensus table + 6×6 agreement matrix across all models |
| Activity History | Per-user rated activity log with star ratings |
| Nearest-User Matching | Identifies the closest synthetic user to a custom DASS profile |

---

## Recommendation Algorithms

### 1. MindMatch Hybrid *(Primary Model)*
A two-model weighted blend:

```
final_score = α × CBF_score + (1 − α) × SVD_score     (α = 0.5)
```

Combines the interpretability of content-based filtering with the personalisation power of collaborative filtering. This is the default and recommended model.

### 2. SVD Collaborative Filtering
Latent-factor matrix factorisation implemented via the [Surprise](https://surpriselib.com/) library. Decomposes the user–activity interaction matrix into latent embeddings and predicts ratings from the dot product of user and item factors.

### 3. Content-Based Filtering
Constructs a TF-IDF profile for each user from their rated activities (using activity name, category, stress dimension, and duration as features). Unseen activities are ranked by cosine similarity to this profile.

### 4. User-Based Collaborative Filtering
Neighbourhood-based approach: identifies the *k* most similar users (by DASS score profile and rating history) and surfaces activities highly rated by those neighbours but not yet seen by the target user.

### 5. Item-Based Collaborative Filtering
Computes item–item cosine similarity from the ratings matrix. Recommends activities most similar to those the user has already rated positively.

### 6. Neural Collaborative Filtering (NeuMF)
Implements the NeuMF architecture combining:
- **GMF** (Generalised Matrix Factorisation) — linear interaction modelling
- **MLP** branch — non-linear deep interaction modelling

Trained on implicit feedback with negative sampling.

### 7. Popularity Baseline
Ranks activities by Bayesian-average rating across all users. Used as a non-personalised baseline for evaluation comparison.

---

## Dataset

The dataset is synthetically generated to simulate a realistic mental wellness platform.

| Property | Value |
|---|---|
| Users | 1,000 |
| Activities | 60 |
| Interactions (ratings) | ~11,500 |
| Matrix sparsity | ~80% |
| Rating scale | 1–5 stars |
| User attributes | Age group, occupation, DASS-21 scores (Depression / Anxiety / Stress, 0–42 each) |
| Activity attributes | Name, category, duration, target stress dimension, effectiveness level |

**Activity categories:** Mindfulness · Physical · Creative · Social · Rest · Cognitive · Lifestyle

**DASS-21 severity bands:**

| Score Range | Severity |
|---|---|
| Depression 0–9 / Anxiety 0–7 / Stress 0–14 | Normal |
| Depression 10–13 / Anxiety 8–9 / Stress 15–18 | Mild |
| Depression 14–20 / Anxiety 10–14 / Stress 19–25 | Moderate |
| Depression 21–27 / Anxiety 15–19 / Stress 26–33 | Severe |
| Depression 28+ / Anxiety 20+ / Stress 34+ | Extremely Severe |

Data is generated on first run and cached to `backend/data/` as CSV files.

---

## Evaluation Metrics

All models are evaluated on a held-out test set (80/20 train-test split) at **K = 5** and **K = 10**.

| Metric | Description |
|---|---|
| **Precision@K** | Fraction of top-K recommendations that are relevant |
| **Recall@K** | Fraction of relevant items captured in top-K recommendations |
| **NDCG@K** | Normalised Discounted Cumulative Gain — rewards correct items ranked higher |
| **Coverage** | Fraction of the activity catalogue that appears in any recommendation list |
| **Diversity** | Average pairwise category-dissimilarity within recommendation lists |
| **RMSE** | Root Mean Square Error of predicted vs. actual ratings (rating-based models only) |
| **MAE** | Mean Absolute Error of predicted vs. actual ratings (rating-based models only) |

Metrics are available via the **Analysis → Evaluation Metrics** tab in the UI and through the `/api/evaluation` endpoint.

---

## Fairness Analysis

The system audits algorithmic fairness using **Demographic Parity** — measuring whether NDCG@5 is equal across demographic subgroups.

**Groups evaluated:** Age groups (18–25, 26–35, 36–50, 51+) and occupations (Student, Professional, Manager, etc.)

**Disparate Impact Ratio:**
```
DI = NDCG_minority_group / NDCG_majority_group
```
A DI ≥ 0.8 is considered fair under the industry-standard **4/5ths rule**.

Results are available via **Analysis → Fairness Analysis** in the UI.

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| Web framework | Flask (Python) |
| Cross-origin requests | Flask-CORS |
| Data processing | Pandas, NumPy |
| Machine learning | scikit-learn, SciPy |
| Collaborative filtering | Scikit-Surprise (SVD) |
| Deep learning (NeuMF) | Custom PyTorch-style NumPy implementation |
| NLP / LLM | Google Gemini AI (`google-generativeai`) |
| Semantic similarity | `sentence-transformers` (SBERT) |
| Configuration | `python-dotenv` |

### Frontend
| Component | Technology |
|---|---|
| UI | Vanilla HTML5, CSS3, JavaScript (ES2020) |
| Charts | Chart.js 4.4 |
| Fonts | Inter (Google Fonts) |
| Communication | Fetch API (REST) |

---

## Project Structure

```
RS project/
├── backend/
│   ├── app.py                      # Flask API server — all endpoints
│   ├── requirements.txt            # Python dependencies
│   ├── data/
│   │   ├── generate_data.py        # Synthetic dataset generator (1,000 users, 60 activities)
│   │   ├── prepare_dass21.py       # DASS-21 severity scoring utilities
│   │   ├── products.csv            # 60 wellness activities (generated)
│   │   ├── ratings.csv             # ~11,500 user–activity ratings (generated)
│   │   └── users.csv               # 1,000 user profiles with DASS scores (generated)
│   ├── models/
│   │   ├── hybrid.py               # MindMatchHybrid: α×CBF + (1−α)×SVD
│   │   ├── cf_surprise.py          # SVD via Scikit-Surprise
│   │   ├── collaborative_filtering.py  # User-based & Item-based CF
│   │   ├── content_based.py        # TF-IDF + cosine similarity
│   │   ├── matrix_factorization.py # Explicit matrix factorisation
│   │   └── neural_cf.py            # NeuMF: GMF + MLP
│   ├── nlp/
│   │   └── condition_parser.py     # DASS extraction from natural language (Gemini + fallback)
│   └── evaluation/
│       ├── metrics.py              # Precision, Recall, NDCG, Coverage, Diversity, RMSE, MAE
│       └── fairness.py             # Demographic parity evaluation
├── frontend/
│   ├── index.html                  # Single-page application shell
│   ├── css/
│   │   └── style.css               # Full stylesheet
│   └── js/
│       └── app.js                  # Application logic (~1,200 lines)
├── .env                            # Environment variables (API keys — not committed)
└── README.md
```

---

## Installation & Setup

### Prerequisites
- Python 3.9 or higher
- pip
- A modern web browser (Chrome, Firefox, Edge)
- *(Optional)* A Google Gemini API key for NLP parsing

### 1. Clone the repository

```bash
git clone <repository-url>
cd "RS project"
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

> If `GEMINI_API_KEY` is not provided, the NLP parser will fall back to keyword-based DASS extraction automatically.

### 5. Generate the dataset *(automatic)*

The dataset is generated automatically on the first server start if CSV files are not found in `backend/data/`. No manual step is required.

---

## Running the Application

### Step 1 — Start the Flask backend

```bash
cd backend
python app.py
```

The server starts at `http://localhost:5000`. On first run, it generates the dataset and trains all six models (this takes approximately 30–60 seconds).

Expected output:
```
Data loaded: 60 activities | 1000 users | 11500 interactions
Training SVD-CF model…
Training Content-Based model…
Training MindMatch Hybrid…
...
 * Running on http://127.0.0.1:5000
```

### Step 2 — Open the frontend

Open `frontend/index.html` directly in a browser, or serve it with any static file server:

```bash
# Using Python's built-in server (from the frontend/ directory)
cd frontend
python -m http.server 8080
```

Then navigate to `http://localhost:8080`.

---

## API Reference

All endpoints are prefixed with `http://localhost:5000/api`.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/users` | List all 1,000 user profiles |
| `GET` | `/stats` | Dataset statistics (users, activities, ratings, sparsity) |
| `GET` | `/user/{id}/history` | Activity history for a specific user |
| `GET` | `/recommend/hybrid/{id}` | MindMatch Hybrid recommendations |
| `GET` | `/recommend/svd-cf/{id}` | SVD Collaborative Filtering recommendations |
| `GET` | `/recommend/content-based/{id}` | Content-Based recommendations |
| `GET` | `/recommend/user-cf/{id}` | User-Based CF recommendations |
| `GET` | `/recommend/item-cf/{id}` | Item-Based CF recommendations |
| `GET` | `/recommend/neural-cf/{id}` | Neural CF (NeuMF) recommendations |
| `GET` | `/recommend/popular/{id}` | Popularity baseline recommendations |
| `GET` | `/similar/{product_id}` | Similar activities (content + collaborative) |
| `POST` | `/nlp/parse-condition` | Parse DASS scores from free-text input |
| `POST` | `/recommend/by-condition` | Recommend activities from DASS scores |
| `GET` | `/evaluation` | Offline evaluation metrics for all models |
| `GET` | `/fairness` | Fairness analysis (demographic parity) |
| `GET` | `/compare-all/{id}` | Top-10 recommendations from all 6 models |
