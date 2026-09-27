# Deliverable 5: Combined Explicit and Implicit ALS

## Score Transformation

The combined ALS model trains on one score column, `combined_score`, built from rating, read, and review signals.

For user `u` and book `i`:

```text
combined_score_ui =
    r_ui             if r_ui > 0
    review_weight    if r_ui = 0 and review_ui = 1
    read_weight      if r_ui = 0 and review_ui = 0 and read_ui = 1
    excluded         otherwise
```

Final weights:

```text
read_weight = 1.0
review_weight = 2.0
```

So the final transformation is:

```text
combined_score_ui =
    r_ui    if r_ui > 0
    2.0     if r_ui = 0 and review_ui = 1
    1.0     if r_ui = 0 and review_ui = 0 and read_ui = 1
    excluded otherwise
```

Ratings are kept unchanged. Read/review signals are used only when no rating is available. This avoids treating a low-rated book as a strong positive just because it was read or reviewed.

ALS was run with:

```text
implicitPrefs = False
ratingCol = combined_score
top-k evaluation = 100
```

## Sampling

The final experiment uses the allowed 3% user sample. Sampling is deterministic and user-based: selected users are kept with all of their interactions. Users with fewer than 5 interactions are removed.

## Splitting

For each sampled user, interactions are randomly ordered with seed `42` and split approximately:

```text
60% train / 20% validation / 20% test
```

This keeps validation and test users connected to training data, which is necessary for ALS recommendations.

## Cluster-Conscious Design

The implementation is staged to reduce reruns on Dataproc:

1. Save the 3% sample and split once.
2. Save each trained ALS model to HDFS.
3. Save validation and test metrics as CSV.
4. Reuse saved models when only evaluation needs to be rerun.
5. Keep the grid small instead of running a large hyperparameter search.

This design follows the assignment update: sample when needed, save intermediate outputs, reduce the grid, and avoid unnecessary recomputation.

## Process Justification

I first used the completed 1% runs to choose the combined-score weights. Then I carried the best weights to the 3% sample and tuned only rank and regularization. This keeps the final process goal-oriented: choose a reasonable signal transformation, then spend the larger 3% run on the ALS parameters most likely to affect ranking quality.

## Grid Search

The 1% weight grid used `recPool = 250`.

| Sample | Rank | RegParam | Read weight | Review weight | MAP | NDCG@100 | P@100 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% | 50 | 0.1 | 1.0 | 2.0 | 0.000538 | 0.003113 | 0.000702 | 0.008151 |
| 1% | 50 | 0.1 | 2.0 | 3.0 | 0.000526 | 0.002600 | 0.000548 | 0.006694 |
| 1% | 50 | 0.1 | 3.0 | 4.0 | 0.000702 | 0.002471 | 0.000451 | 0.005923 |

The 3% rank/regularization grid reused `read_weight = 1.0`, `review_weight = 2.0`, and used `recPool = 500`.

| Sample | Rank | RegParam | Read weight | Review weight | MAP | NDCG@100 | P@100 | R@100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3% | 100 | 0.01 | 1.0 | 2.0 | 0.001654 | 0.007444 | 0.001874 | 0.017050 |
| 3% | 100 | 0.1 | 1.0 | 2.0 | 0.000368 | 0.001925 | 0.000384 | 0.005151 |
| 3% | 50 | 0.1 | 1.0 | 2.0 | 0.000311 | 0.001623 | 0.000357 | 0.004057 |

Final selected configuration:

```text
rank = 100
regParam = 0.01
read_weight = 1.0
review_weight = 2.0
recPool = 500
maxIter = 5
```

## Final Results

Primary relevance label: `rating >= 4`. Secondary relevance label: `is_read = 1`.

| Split | Label | MAP | NDCG@100 | P@100 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | liked (`rating >= 4`) | 0.001654 | 0.007444 | 0.001874 | 0.017050 |
| Test | liked (`rating >= 4`) | 0.001751 | 0.007562 | 0.001859 | 0.017040 |
| Test | read (`is_read = 1`) | 0.001616 | 0.007505 | 0.002169 | 0.015577 |

## Analysis Compared With Baseline

Popularity baseline values are from the saved screenshot.

| Split | Label | MAP | NDCG@100 | P@100 | R@100 |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | `is_read = 1` | 0.030871 | 0.105842 | 0.027130 | 0.157686 |
| Test | `is_read = 1` | 0.032182 | 0.108916 | 0.027592 | 0.162180 |
| Validation | `rating >= 4` | 0.032777 | 0.104282 | 0.020282 | 0.173023 |
| Test | `rating >= 4` | 0.034330 | 0.107032 | 0.020639 | 0.176481 |

Main comparison on test `rating >= 4`:

| Model | Test MAP | Test NDCG@100 | Test P@100 | Test R@100 |
| --- | ---: | ---: | ---: | ---: |
| Popularity baseline | 0.034330 | 0.107032 | 0.020639 | 0.176481 |
| Explicit ALS |  |  |  |  |
| Implicit ALS |  |  |  |  |
| Combined ALS | 0.001751 | 0.007562 | 0.001859 | 0.017040 |

The combined ALS model improved after moving from the 1% run to the 3% tuned run, but it still underperformed the popularity baseline. The likely reason is that Goodreads held-out positives include many globally popular books, while ALS depends on enough user-item overlap in the sampled graph.

## Limitations and Potential Fixes

- The final model uses 3% of users, not the full data. Full-data training could improve overlap, but would require more cluster resources.
- `recPool = 500` may still be too small after filtering seen books. Increasing it to 1000 could improve recall if the cluster can handle it.
- The grid was intentionally sparse. More rank, regularization, and iteration values could improve results.
- The score transformation is simple. Another approach is to treat rating as the explicit preference signal and use `is_read` and `is_reviewed` as confidence modifiers.
- A hybrid model that blends ALS with popularity may work better because the popularity baseline is very strong.


