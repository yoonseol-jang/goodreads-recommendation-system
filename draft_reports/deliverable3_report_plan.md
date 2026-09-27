# Plan: Deliverable 3 Report Content (Explicit-Feedback ALS)

## 1. Model objective

- Train a latent-factor recommender using Spark ALS.
- Use **explicit feedback only**: the numerical Goodreads `rating`.
- Exclude `rating = 0` rows from ALS training because they do not contain an explicit numerical preference.
- Do **not** use `is_read` or `is_reviewed` as model inputs for this deliverable.

## 2. Train/validation/test split

- Experiments were run on the deterministic 1% user-based sample.
- Users were sampled by hashing `user_id`; all interactions for sampled users were retained.
- Users with fewer than 5 interactions were removed before splitting.
- For each remaining user, interactions were randomly ordered with seed 42 and split:
  - 60% train
  - 20% validation
  - 20% test
- During recommendation, books already present in the user's training history were removed from the recommendation list.

## 3. Training setup

- ALS training data:
  - `userCol = user_id`
  - `itemCol = book_id`
  - `ratingCol = rating`
  - train rows restricted to `rating > 0`
- Spark ALS was run with `coldStartStrategy="drop"`.
- Each `(rank, regParam)` setting was submitted as a separate Spark job on Dataproc. Each job wrote its validation metrics and saved model to HDFS. A finalizer script aggregated the validation grid and selected the best model.

## 4. Evaluation setup

- Generate top-100 recommendations for each evaluable user.
- Relevant held-out books were defined as books with `rating >= 4`.
- Model selection metric: validation `NDCG@100`.
- Ranking metrics reported:
  - MAP
  - NDCG@100
  - Precision@100
  - Recall@100
- RMSE was also reported as a secondary rating-prediction metric, but ranking metrics were the main evaluation criterion.

## 5. Hyperparameters considered

The validation search covered selected combinations across:

| Hyperparameter | Values considered |
| --- | --- |
| `rank` | 5, 20, 50, 100 |
| `regParam` | 0.001, 0.01, 0.1, 0.5 |

13 of the 16 cells were evaluated (partial grid). The best model was selected by validation `NDCG@100`.

Validation `NDCG@100` across the grid (— = not evaluated, **bold** = best):

| rank \ regParam | 0.001 | 0.01 | 0.1 | 0.5 |
| ---: | ---: | ---: | ---: | ---: |
| 5   | —        | 0.000172 | 0.000168 | 0.000154 |
| 20  | —        | 0.002061 | 0.001441 | 0.000834 |
| 50  | 0.005863 | 0.005865 | 0.002249 | 0.000809 |
| 100 | 0.014431 | **0.014463** | 0.000221 | —        |

Observations:
- Increasing `rank` produces the largest improvement: NDCG@100 climbs roughly two orders of magnitude from `rank=5` to `rank=100`.
- At `rank=100`, the score is robust across `regParam ∈ {0.001, 0.01}` but collapses at `regParam=0.1`, suggesting strong regularization erases the latent structure at high rank.
- The grid is wide enough to produce clearly observable score differences across both hyperparameters.

Best configuration:

| rank | regParam |
| ---: | ---: |
| 100 | 0.01 |

## 6. Validation performance

Best explicit ALS model on validation:

| Metric | Value |
| --- | ---: |
| MAP | 0.002810 |
| NDCG@100 | 0.014463 |
| Precision@100 | 0.003379 |
| Recall@100 | 0.034114 |
| RMSE | 1.932319 |

## 7. Test performance

Final test performance for the selected model:

| Metric | Value |
| --- | ---: |
| MAP | 0.003041 |
| NDCG@100 | 0.015756 |
| Precision@100 | 0.003624 |
| Recall@100 | 0.036810 |
| RMSE | 1.927922 |

## 8. Comparison to popularity baseline

For comparison with explicit ALS, use the popularity baseline evaluated with the same `rating >= 4` relevance label.

| Model | MAP | NDCG@100 | Precision@100 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| Popularity baseline | 0.034330 | 0.107032 | 0.020639 | 0.176481 |
| Explicit ALS | 0.003041 | 0.015756 | 0.003624 | 0.036810 |

The explicit ALS model underperformed the popularity baseline on ranking metrics. A likely explanation is that explicit ratings are sparse in the 1% sample, while globally popular books are a strong signal in Goodreads recommendation.

## 9. Summary sentence

The explicit-feedback ALS model satisfied the latent-factor modeling requirement and improved with larger rank, but its best configuration (`rank=100`, `regParam=0.01`) did not outperform the popularity baseline on top-100 ranking metrics.
