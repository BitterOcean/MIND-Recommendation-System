# MIND-Recommendation-System

### Content-Based, Collaborative & Hybrid Models on MIND Dataset

A comparative study of **recommender system strategies** for news articles using the **MIND dataset**, focusing on both **accuracy** and **beyond-accuracy metrics**.

---

## Table of Contents

* [1. Introduction](#1-introduction)
* [2. Problem Statement](#2-problem-statement)
* [3. Approach](#3-approach)
* [4. Dataset & EDA](#4-dataset--eda)
* [5. Methods](#5-methods)
* [6. Evaluation Metrics](#6-evaluation-metrics)
* [7. Results](#7-results)
* [8. Discussion](#8-discussion)
* [9. Conclusion](#9-conclusion)
* [10. Future Work](#10-future-work)

---

# 1. Introduction

Recommender systems are a core component of modern digital platforms, powering personalized experiences across news, e-commerce, and media.

This project explores and compares three major recommendation strategies:

* **Content-Based Filtering (CBF)**
* **Collaborative Filtering (CF)**
* **Hybrid Models**

The goal is to evaluate how these approaches perform in a **news recommendation setting** using the **MIND dataset**, a widely used benchmark in the field .

[⬆️ Back to Top](#table-of-contents)

---

# 2. Problem Statement

Designing an effective news recommender involves several challenges:

### Challenges

* **Data Sparsity** → Most users interact with very few articles
* **Cold-Start Problem** → Many articles have no interaction history
* **Popularity Bias** → Popular items dominate recommendations
* **Trade-offs** → Accuracy vs diversity vs novelty

### Goal

Build and compare multiple recommender systems to answer:

> How do Content-Based, Collaborative, and Hybrid approaches perform in terms of both accuracy and beyond-accuracy metrics?

[⬆️ Back to Top](#table-of-contents)

---

# 3. Approach

We implemented a **multi-model comparison pipeline**:

## Models Compared

* Popularity Baseline
* Content-Based Recommender (TF-IDF)
* Collaborative Filtering (SVD)
* Weighted Hybrid Model

## Strategy

1. Analyze dataset characteristics (EDA)
2. Implement each model independently
3. Evaluate using multiple metrics
4. Compare across datasets (MIND Small & Large)

[⬆️ Back to Top](#table-of-contents)

---

# 4. Dataset & EDA

## Dataset: MIND (Microsoft News Dataset)

* ~1M users
* ~160k articles
* ~2.2M sessions 

Contains:

* Article metadata (title, abstract, category)
* User interaction logs (clicks, impressions)

## Key Insights from EDA

* **Long-tail distribution** → Few articles get most clicks
* **Category imbalance** → News & sports dominate
* **Short text features** → Titles are short, abstracts vary
* **Sparse interactions** → Most users have limited history

📊 Example:

* ~85% of articles receive zero clicks in training 

👉 These insights directly influenced model design:

* Use CBF for cold-start
* Expect CF to struggle with sparsity
* Combine models via hybrid approach

[⬆️ Back to Top](#table-of-contents)

---

# 5. Methods

## 5.1 Popularity Baseline

* Scores items by total click count
* Non-personalized
* Strong baseline due to popularity bias

---

## 5.2 Content-Based Filtering (CBF)

* TF-IDF representation of:

  * Title
  * Abstract
  * Category
  * Subcategory
* User profile:

  * Recency-weighted history
* Scoring:

  * Cosine similarity

✔ Handles cold-start
✔ Strong performance in sparse data

---

## 5.3 Collaborative Filtering (CF)

* User-item interaction matrix
* Factorized using **Truncated SVD (32 dimensions)**
* User profile from item embeddings

✔ Captures behavioral patterns
✘ Cannot handle unseen items (cold-start)

---

## 5.4 Hybrid Model

Combines CF and CBF scores:

$$
S_{final} = \omega_{cf} \cdot norm(S_{cf}) + \omega_{cbf} \cdot norm(S_{cbf})
$$

* Scores normalized (min-max)
* Best weights:

  * **CF = 0.1**
  * **CBF = 0.9**

✔ Combines strengths of both models
✔ Improves overall performance

[⬆️ Back to Top](#table-of-contents)

---

# 6. Evaluation Metrics

## Accuracy Metrics

* **AUC** → overall ranking quality
* **MRR** → rewards early correct predictions
* **nDCG@5 / nDCG@10** → ranking quality at top positions

## Beyond-Accuracy Metrics

* **Novelty** → recommends less popular items
* **Diversity (category/subcategory)** → variety of recommendations

👉 Important insight:

> Accuracy alone is not enough — recommendation quality is multi-dimensional.

[⬆️ Back to Top](#table-of-contents)

---

# 7. Results

## MIND Small Dataset

| Model      | AUC        | MRR        | nDCG@10    | Novelty |
| ---------- | ---------- | ---------- | ---------- | ------- |
| Baseline   | 0.5318     | 0.2671     | 0.3098     | 14.04   |
| CF         | 0.5429     | 0.2632     | 0.3088     | 14.58   |
| CBF        | 0.6073     | 0.3306     | 0.3718     | 16.03   |
| **Hybrid** | **0.6126** | **0.3310** | **0.3731** | 15.80   |

---

## MIND Large Dataset

| Model      | AUC        | MRR        | nDCG@10    |
| ---------- | ---------- | ---------- | ---------- |
| Baseline   | 0.5385     | 0.2618     | 0.3079     |
| CF         | 0.5541     | 0.2795     | 0.3223     |
| CBF        | 0.6059     | 0.3300     | 0.3711     |
| **Hybrid** | **0.6084** | **0.3272** | **0.3692** |

📌 Key finding:

> Hybrid > Content-Based > Collaborative > Baseline 

[⬆️ Back to Top](#table-of-contents)

---

# 8. Discussion

### Key Insights

* **CBF is strongest in sparse settings**
* **CF improves with more data**
* **Hybrid provides consistent gains**

### Trade-offs

* High accuracy → lower diversity
* High novelty → less popular items

Hybrid balances these competing objectives.

[⬆️ Back to Top](#table-of-contents)

---

# 9. Conclusion

* Content-based filtering is the **strongest standalone model**
* Collaborative filtering requires **large-scale data**
* Hybrid models provide **best overall performance**

Most importantly:

> No single metric fully captures recommendation quality.

[⬆️ Back to Top](#table-of-contents)

---

# 10. Future Work

* Neural Collaborative Filtering (NCF)
* Transformer-based embeddings (BERT / LLMs)
* Temporal-aware recommendations
* Better hybrid fusion strategies

---

## 🔗 Resources

* 📄 Full Report: https://github.com/BitterOcean/MIND-Recommendation-System/blob/main/Report.pdf
* 📊 Presentation: https://github.com/BitterOcean/MIND-Recommendation-System/blob/main/Presentation.pdf
* 💻 Code: https://github.com/BitterOcean/MIND-Recommendation-System

---
