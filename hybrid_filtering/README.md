# Hybrid Filtering

This folder contains a weighted hybrid recommender that combines:
- collaborative filtering from `collaborative_filtering`
- content-based filtering from `content_based_filtering`

The current implementation fuses per-impression scores from the two models after
normalizing them onto a comparable scale. A baseline scorer can be added later as
an additional weighted component without changing the overall interface.

## Overview

Model:
- Weighted score fusion
- Collaborative filtering score from latent item embeddings
- Content-based score from TF-IDF user profiles
- Per-impression normalization before fusion

Fusion:
- `hybrid_score = cf_weight * norm(cf_score) + cbf_weight * norm(cbf_score)`

Normalization options:
- `minmax`
- `zscore`
- `rank`
- `none`

Evaluation metrics:
- AUC
- MRR
- nDCG@5
- nDCG@10
- Coverage@10

## Run

From the project root:

```bash
python -m hybrid_filtering
```

Useful options:

```bash
python -m hybrid_filtering --cf-weight 0.3 --cbf-weight 0.7
python -m hybrid_filtering --normalization rank
python -m hybrid_filtering --tune-weights --weight-grid-step 0.1 --tuning-metric ndcg_at_10
python -m hybrid_filtering --cf-components 64 --cbf-max-features 75000
python -m hybrid_filtering --max-eval-sessions 5000
python -m hybrid_filtering --model-output hybrid_filtering/artifacts/weighted_hybrid_model.json
```

When `--tune-weights` is enabled, the command evaluates weight pairs that satisfy:
- `cf_weight + cbf_weight = 1`

The best pair is selected on the chosen validation metric, then saved as the final
hybrid model.

## Saved model

The fitted hybrid model is stored as:
- a metadata `.json`
- a saved collaborative-filtering artifact
- a saved content-based artifact set

and can be loaded with:

```python
from hybrid_filtering.models import WeightedHybridRecommender

model = WeightedHybridRecommender.load(
    "hybrid_filtering/artifacts/weighted_hybrid_model.json"
)
```
