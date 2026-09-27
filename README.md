# Goodreads Recommendation System & Market Segmentation

Large-scale book recommender system and market segmentation pipeline built on 223M user-book interactions using Apache Spark on GCP Dataproc.

**Authors:** Yoonseol Jang, David Lee, Donggyu Kim — DS-GA 1004 Big Data, Spring 2026

## Summary

- **Dataset:** Goodreads interactions — 876K users, 2.4M books, 223M interactions
- **Infrastructure:** GCP Dataproc (HDFS), Apache Spark, Parquet
- **Models:** Popularity baseline, Explicit ALS, Implicit ALS, Combined ALS
- **Best model:** Implicit ALS — 76% relative improvement in MAP over popularity baseline
- **Market segmentation:** MinHash LSH over binary user-reader sets; top pair Jaccard similarity 0.9714

## Pipeline

```
src/
  01_convert_to_parquet.py          CSV → Parquet (schema-explicit, no inferSchema)
  02_check_parquet.py               Validate conversion
  03_make_user_sample.py            Deterministic 3% user sample via xxhash64
  04_split_sample.py                Per-user 60/20/20 train/val/test split
  05_popularity_baseline_sample.py  Global popularity ranker
  06_check_popularity_results.py
  07_market_segmentation_minhash_sample.py  MinHash LSH similarity search
  08_check_market_segmentation_results.py
  09_explicit_als_sample.py         Explicit-feedback ALS (rating signal)
  10_finalize_explicit_als.py
  11_implicit_als_sample.py         Implicit-feedback ALS (is_read + is_reviewed)
  12_check_implicit_als_results.py
  13_combined_als_sample.py         Combined explicit + implicit score
  14_finalize_combined_als.py
```

## Results

### Popularity Baseline (is_read labels, 3% test)

| MAP | NDCG@100 | P@100 | R@100 |
|---|---|---|---|
| 0.0322 | 0.1087 | 0.0276 | 0.1601 |

### Explicit ALS — rank=100, regParam=0.01 (rating≥4 labels, 3% test)

| MAP | NDCG@100 | P@100 | R@100 |
|---|---|---|---|
| 0.00158 | 0.00678 | 0.00158 | 0.01583 |

Underperformed popularity baseline — sparse explicit ratings make global popularity hard to beat at this scale.

### Implicit ALS — rank=100, regParam=0.1, alpha=15.0 (is_read labels, 3% test)

| MAP | NDCG@100 | P@100 | R@100 |
|---|---|---|---|
| 0.05664 | 0.1722 | 0.04585 | 0.25093 |

**76% relative improvement in MAP**, 58% in NDCG@100, 66% in P@100 over popularity baseline.

### Combined ALS (rating + is_read + is_reviewed, 3% test)

Combined score: rating if rated; 2.0 if read+reviewed; 1.0 if read only. Trained in explicit mode with `ratingCol=combined_score`.

### Market Segmentation — MinHash LSH (5 hash tables, Jaccard threshold 0.8)

Top book pair Jaccard similarity: **0.9714** (34 shared readers out of 35). Top 100 pairs ranged from 0.97 to 0.86, identifying books occupying similar market positions. Brute-force O(N²) search replaced by MinHash candidate generation + exact Jaccard verification in a single distributed pass.

## Key Design Decisions

- **Deterministic sampling:** `xxhash64(user_id) % 100 < 3` for reproducible 3% sample; same rule at 1% for hyperparameter screens
- **Per-user splits:** interactions split within each user (60/20/20), not across users — eliminates cold-start from eval cohort
- **Cluster-conscious staging:** 3% sample and splits written to HDFS once; trained ALS models persisted and reused across eval runs
- **Implicit signal construction:** `implicit_score = is_read + 2×is_reviewed` — reviewed books weighted 2× to reflect higher engagement signal vs. passive consumption

## Tech Stack

`Apache Spark` `PySpark` `GCP Dataproc` `HDFS` `Parquet` `MinHash LSH` `ALS` `RankingMetrics`
