# H&M Personalized Fashion Recommendations

A complete two-stage recommendation pipeline built on the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) Kaggle competition dataset (31.8M transactions, 1.37M customers, 105K articles).

**MAP@12 = 0.0588 · NDCG@12 = 0.0821**

> Evaluated on held-out test week (Sep 16-22), boundary artifact users excluded.

---

## Pipeline Overview

```
Transactions (31.8M)
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 1 — Candidate Retrieval (~320 per user)  │
│                                                 │
│  Rule-based (300 slots):                        │
│  · s1  — postal top-1000 products (primary)     │
│           segment × age fallback                │
│  · s2a — purchase history recency (15 slots)    │
│  · s2b — colour variants (15 slots)             │
│  · s3  — co-occurrence / bought together        │
│           (30 slots)                            │
│                                                 │
│  ALS collaborative filtering (+50 slots)        │
└─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────┐
│  Stage 2 — LightGBM LambdaRank Reranker         │
│                                                 │
│  36 features:                                   │
│  · Recency / frequency (repurchase signals)     │
│  · Item popularity & momentum (1w / 2w / 4w)    │
│  · Price ratio                                  │
│  · Co-occurrence count                          │
│  · Category / segment / colour in a history     │
│  · ALS user score + CF similarity               │
│                                                 │
│  Tuned with Optuna (50 trials)                  │
└─────────────────────────────────────────────────┘
        │
        ▼
   Top-12 recommendations per user
```

---

## Results

| Stage | Metric | Value |
|-------|--------|-------|
| Rule-based retrieval | Recall@300 | 0.2747 |
| + ALS candidates | Recall@300 | 0.2818 |
| LightGBM reranker | MAP@12 | **0.0588** |
| LightGBM reranker | NDCG@12 | **0.0821** |
| Global top-12 baseline | MAP@12 | 0.0096 |

---

## Repository Structure

```
hm-recommendations/
│
├── notebooks/
│   ├── nb1_eda.ipynb               # Exploratory data analysis
│   ├── nb2a_rules.ipynb            # Rule-based candidate generation
│   ├── nb2b_als_cf.ipynb           # ALS collaborative filtering
│   └── nb3_reranker.ipynb          # LightGBM reranker + Optuna tuning
│
├── requirements.txt
└── README.md
```

---

## Notebooks

### `nb1_eda.ipynb` — Exploratory Data Analysis
Deep analysis of 31.8M transactions covering:
- Item popularity distribution and long-tail behaviour
- Popularity stability: how fast top items churn (35% week-over-week overlap)
- Seasonal patterns (swimwear, coats, holiday items)
- User segmentation by gender, age, postal code
- Repurchase analysis: 14.1% exact repurchase rate, colour exploration 5× more common than size switching
- Price distribution and discount hunter segmentation
- Cold-start characterisation (8.1% of test users have zero history)

Key finding: **last-week popularity window outperforms longer windows** — top-1000 items from 1 week ago cover 49.2% of next-week transactions vs 36.5% from a 6-week window.

### `nb2a_rules.ipynb` — Rule-Based Candidate Generation
Generates ~300 candidates per user using 4 strategies:
- **s1** — postal code top-1000 products (99.8% transaction coverage), with segment × age fallback
- **s2a** — user purchase history sorted by recency (15 slots)
- **s2b** — colour variants of purchased items ranked by popularity (15 slots)
- **s3** — items frequently bought together via co-occurrence matrix (30 slots)

Achieves **Recall@300 = 0.2747**.

### `nb2b_als_cf.ipynb` — ALS Collaborative Filtering
Trains an Alternating Least Squares model (64 factors, 20 iterations) on the last 6 weeks of transactions. Two outputs:
- **50 ALS candidates** per user added to rule-based pool → Recall@300 = 0.2818
- **CF score features** (`als_user_score`, `als_item_cf_score`) used by the reranker

### `nb3_reranker.ipynb` — LightGBM Reranker
Trains a LightGBM LambdaRank model on the combined candidate pool with 36 engineered features. Key design decisions:
- Labels from test-week purchases (Sep 16-22)
- Boundary artifact users removed (>50% item overlap between last train week and test week)
- 80/20 user split: Optuna tunes on 20% held-out val users
- Final model trained on all users with best Optuna params
- Feature selection via forward selection (documented in notebook)

---

## Data

The dataset is from the [Kaggle H&M competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) and is not included in this repository. Download it from Kaggle and place it at:

```
/content/drive/MyDrive/HM Fashion/
├── processed/
│   ├── transactions.csv
│   ├── articles.csv
│   └── customers.csv
└── images/
    └── {article_id[:3]}/{article_id}.jpg
```

---

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in order: `nb1` → `nb2a` → `nb2b` → `nb3`

---

## Requirements

```
pandas
numpy
lightgbm
implicit          # ALS
optuna
shap
matplotlib
scipy
scikit-learn
```
