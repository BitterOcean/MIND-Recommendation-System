# Collaborative Filtering

This folder contains the collaborative-filtering implementation for the project.

## Overview

Model:
- Implicit-feedback collaborative filtering
- Item embeddings learned with truncated SVD on the item-user interaction matrix
- User profile built from the articles in the current reading history
- Candidate articles ranked by cosine similarity to that profile

Training signals:
- Click history from `behaviors.tsv`
- Positive clicks inside the train impressions

Evaluation metrics:
- AUC
- MRR
- nDCG@5
- nDCG@10
- Coverage@10

## Run

From the project root:

```bash
python -m collaborative_filtering
```

Useful options:

```bash
python -m collaborative_filtering --components 32 --history-decay 0.9
python -m collaborative_filtering --max-eval-sessions 5000
python -m collaborative_filtering --train-dir data/MINDsmall_train --eval-dir data/MINDsmall_dev
python -m collaborative_filtering --model-output collaborative_filtering/artifacts/mind_small_cf.npz
```

By default, the fitted model is saved to `collaborative_filtering/artifacts/latent_item_cf_model.npz`.

## Saved model

The fitted model is stored as a compressed `.npz` artifact and can be loaded with:

```python
from collaborative_filtering.models import LatentItemCFRecommender

model = LatentItemCFRecommender.load(
    "collaborative_filtering/artifacts/latent_item_cf_model.npz"
)
```

## Current full-split result

Running on local `MINDsmall_train` and `MINDsmall_dev` produced:

```json
{
  "auc": 0.5429,
  "mrr": 0.2632,
  "ndcg_at_5": 0.2473,
  "ndcg_at_10": 0.3088,
  "coverage_at_10": 0.4343
}
```
