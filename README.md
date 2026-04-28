# 📰 News Recommendation System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![AUC](https://img.shields.io/badge/AUC-61.26%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green.svg)

### Content-Based, Collaborative & Hybrid Models on MIND Dataset

A comparative study of recommender system strategies for news articles using the **MIND dataset**, focusing on both **accuracy** and **beyond-accuracy metrics**.

---

## 📚 Table of Contents

* [1. Introduction](#1-introduction)
* [2. Problem Statement](#2-problem-statement)
* [3. Approach](#3-approach)
* [4. Dataset & EDA](#4-dataset--eda)
* [5. Methods](#5-methods)
* [6. Evaluation Metrics](#6-evaluation-metrics)
* [7. Results](#7-results)
* [8. Discussion](#8-discussion)
* [9. Conclusion](#9-conclusion)
* [10. How to Run](#10-how-to-run)
* [11. Future Work](#11-future-work)

---

# 1. Introduction

Recommender systems are a core component of modern digital platforms.

This project compares:

* **Content-Based Filtering (CBF)**
* **Collaborative Filtering (CF)**
* **Hybrid Models**

on the **MIND dataset**, a widely used benchmark in news recommendation.

---

# 2. Problem Statement

### Challenges

* Data sparsity
* Cold-start problem
* Popularity bias
* Trade-offs between accuracy, diversity, and novelty

### Goal

Evaluate and compare recommender models across multiple metrics.

---

# 3. Approach

Models implemented:

* Popularity Baseline
* Content-Based (TF-IDF)
* Collaborative Filtering (SVD)
* Hybrid Model

Pipeline:

1. EDA
2. Model implementation
3. Evaluation
4. Comparison

---

# 4. Dataset & EDA

## 📊 Dataset: MIND

🔗 https://msnews.github.io/

* ~1M users
* ~160k articles
* ~2.2M sessions

### Key Insights

* Long-tail distribution
* Category imbalance
* Sparse user interactions
* Short textual features

👉 Leads to:

* Strong CBF performance
* Weak CF in small data
* Need for hybrid model

---

# 5. Methods

## Baseline

* Popularity-based ranking

## Content-Based (CBF)

* TF-IDF on title, abstract, category
* Cosine similarity
* Recency-weighted user profile

## Collaborative Filtering (CF)

* User-item matrix
* Truncated SVD (32 dims)
* Embedding-based similarity

## Hybrid

[
S = \omega_{cf} \cdot norm(S_{cf}) + \omega_{cbf} \cdot norm(S_{cbf})
]

Best weights:

* CF = 0.1
* CBF = 0.9

---

# 6. Evaluation Metrics

### Accuracy

* AUC
* MRR
* nDCG@5 / nDCG@10

### Beyond-Accuracy

* Novelty
* Diversity

---

# 7. Results

## MIND Small

| Model      | AUC        | MRR        | nDCG@10    | Novelty |
| ---------- | ---------- | ---------- | ---------- | ------- |
| Baseline   | 0.5318     | 0.2671     | 0.3098     | 14.04   |
| CF         | 0.5429     | 0.2632     | 0.3088     | 14.58   |
| CBF        | 0.6073     | 0.3306     | 0.3718     | 16.03   |
| **Hybrid** | **0.6126** | **0.3310** | **0.3731** | 15.80   |

---

## MIND Large

| Model      | AUC        | MRR        | nDCG@10    |
| ---------- | ---------- | ---------- | ---------- |
| Baseline   | 0.5385     | 0.2618     | 0.3079     |
| CF         | 0.5541     | 0.2795     | 0.3223     |
| CBF        | 0.6059     | 0.3300     | 0.3711     |
| **Hybrid** | **0.6084** | **0.3272** | **0.3692** |

📌 Hybrid performs best overall.

---

# 8. Discussion

* CBF dominates in sparse data
* CF improves with scale
* Hybrid balances both

Trade-offs:

* Accuracy vs diversity
* Novelty vs popularity

---

# 9. Conclusion

* CBF = strongest standalone
* CF = data-dependent
* Hybrid = best overall

---

# 10. How to Run

## 📥 1. Download Dataset

Download from:

👉 https://msnews.github.io/

Files needed:

* MINDsmall_train.zip
* MINDsmall_dev.zip

---

## 📂 2. Extract Dataset

```bash
data/
├── MINDsmall_train/
│   ├── behaviors.tsv
│   ├── news.tsv
│   ├── entity_embedding.vec
│   └── relation_embedding.vec
│
├── MINDsmall_dev/
│   ├── behaviors.tsv
│   ├── news.tsv
│   ├── entity_embedding.vec
│   └── relation_embedding.vec
```

### File Descriptions

* **behaviors.tsv** → user clicks & impressions
* **news.tsv** → article metadata
* **entity_embedding.vec** → entity embeddings
* **relation_embedding.vec** → relation embeddings

⚠️ Only `behaviors.tsv` and `news.tsv` are used in this project.

---

## 📦 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ 4. Run

```bash
python run_all.py
```

---

## ⚙️ Optional

```bash
python run_all.py --max-eval-sessions 5000
```

---

## 💾 Save Results

```bash
python run_all.py --json-output results.json
```

---

## 📁 Project Structure

```bash
project-root/
├── run_all.py
├── requirements.txt
├── data/
├── baseline/
├── collaborative_filtering/
├── content_based_filtering/
├── hybrid_filtering/
```

---

# 11. Future Work

* Neural CF
* Transformer embeddings
* Temporal modeling
* Better hybrid strategies

---

## 🔗 Resources

* 📄 Full Report: https://github.com/BitterOcean/MIND-Recommendation-System/blob/main/Report.pdf
* 📊 Presentation: https://github.com/BitterOcean/MIND-Recommendation-System/blob/main/Presentation.pdf
* 💻 Code: https://github.com/BitterOcean/MIND-Recommendation-System

---
