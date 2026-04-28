# Content-Based Filtering

This folder contains a content-based filtering implementation for the project.
The structure mirrors the collaborative-filtering package so the two models can
be evaluated and combined more easily in a later hybrid stage.

## Overview

Model:
- User-profile content-based filtering
- TF-IDF features built from `title + abstract + category + subcategory`
- User profile built from the articles in the current reading history
- Candidate articles ranked by cosine-style TF-IDF similarity to that profile

Training signals:
- News content from `news.tsv`
- User click history from `behaviors.tsv`

Evaluation metrics:
- AUC
- MRR
- nDCG@5
- nDCG@10
- Coverage@10

## Run

From the project root:

```bash
python -m content_based_filtering
```

Useful options:

```bash
python -m content_based_filtering --history-decay 0.9
python -m content_based_filtering --max-features 50000 --min-df 2 --max-df 0.95
python -m content_based_filtering --max-eval-sessions 5000
python -m content_based_filtering --train-dir data/MINDsmall_train --eval-dir data/MINDsmall_dev
python -m content_based_filtering --model-output content_based_filtering/artifacts/tfidf_cbf_model.json
```

## Saved model

The fitted model is stored as:
- a metadata `.json`
- a sparse TF-IDF matrix `.npz`
- a serialized vectorizer `.pkl`

and can be loaded with:

```python
from content_based_filtering.models import TfidfContentBasedRecommender

model = TfidfContentBasedRecommender.load(
    "content_based_filtering/artifacts/tfidf_cbf_model.json"
)
```

## Current full-split result

Running on local `MINDsmall_train` and `MINDsmall_dev` produced:

```json
{
    "auc": 0.6072362195183464,
    "mrr": 0.3306399555533331,
    "ndcg_at_5": 0.3107397597223671,
    "ndcg_at_10": 0.3717931482672122,
    "coverage_at_10": 0.6768485751536599
  }
```
