from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, row_number, collect_set,
    size, array_intersect, avg,
    sort_array, collect_list, struct, expr
)
from pyspark.sql.window import Window
from pyspark.mllib.evaluation import RankingMetrics


spark = (
    SparkSession.builder
    .appName("Goodreads Popularity Baseline Sample")
    .getOrCreate()
)

base = f"hdfs:///user/{NETID}_nyu_edu/goodreads"

train_path = f"{base}/splits/sample_train.parquet"
val_path = f"{base}/splits/sample_validation.parquet"
test_path = f"{base}/splits/sample_test.parquet"

results_path = f"{base}/results/popularity_baseline_sample"


train = spark.read.parquet(train_path)
validation = spark.read.parquet(val_path)
test = spark.read.parquet(test_path)


# --------------------------------------------------
# Popularity score:
# number of times each book was read in the train set.
# We use is_read = 1 because this baseline recommends books
# based on consumption behavior.
# --------------------------------------------------

popular_books = (
    train
    .where(col("is_read") == 1)
    .groupBy("book_id")
    .agg(count("*").alias("read_count"))
    .orderBy(col("read_count").desc())
)

print("Top popular books:")
popular_books.show(20)


# Keep a deeper candidate pool than 100 because we will remove books
# each user has already seen in train. 1000 gives heavy readers
# enough headroom to still receive 100 recs after the anti-join.
top_candidates = (
    popular_books
    .limit(1000)
    .withColumn("pop_rank", row_number().over(Window.orderBy(col("read_count").desc())))
    .select("book_id", "read_count", "pop_rank")
)


def evaluate_split(eval_df, split_name):
    print(f"Evaluating on {split_name}...")

    # Relevant held-out books:
    # We evaluate whether the model recommends books the user actually read.
    labels = (
        eval_df
        .where(col("is_read") == 1)
        .groupBy("user_id")
        .agg(collect_set("book_id").alias("label_books"))
    )

    # Users we can evaluate
    users = labels.select("user_id")

    # Books already seen in train should not be recommended again
    seen_train = train.select("user_id", "book_id").distinct()

    # Candidate recommendations:
    # every eval user gets global popular books,
    # then remove books already seen in train.
    candidates = (
        users
        .crossJoin(top_candidates)
        .join(seen_train, on=["user_id", "book_id"], how="left_anti")
    )

    # Rank candidates by popularity per user, keep top 100,
    # and collect them as an ORDERED list (sort_array on the (rn, book_id)
    # struct preserves popularity order through the groupBy).
    w = Window.partitionBy("user_id").orderBy(col("pop_rank"))

    recs = (
        candidates
        .withColumn("rn", row_number().over(w))
        .where(col("rn") <= 100)
        .groupBy("user_id")
        .agg(sort_array(collect_list(struct("rn", "book_id"))).alias("ranked"))
        .withColumn("rec_books", expr("transform(ranked, x -> x.book_id)"))
        .drop("ranked")
    )

    eval_table = (
        labels
        .join(recs, on="user_id", how="inner")
    )

    # ----------------------------------------
    # Spark RankingMetrics: MAP, NDCG@100, precision@100
    # ----------------------------------------
    pred_and_labels = (
        eval_table
        .select("rec_books", "label_books")
        .rdd
        .map(lambda r: (list(r["rec_books"]), list(r["label_books"])))
    )

    rm = RankingMetrics(pred_and_labels)
    map_score = rm.meanAveragePrecision
    ndcg_at_100 = rm.ndcgAt(100)
    precision_at_100 = rm.precisionAt(100)

    # ----------------------------------------
    # recall@100: computed by hand because older Spark
    # RankingMetrics does not expose recallAt.
    # ----------------------------------------
    recall_df = (
        eval_table
        .withColumn("hits", size(array_intersect(col("rec_books"), col("label_books"))))
        .withColumn("num_labels", size(col("label_books")))
        .withColumn("recall_at_100", col("hits") / col("num_labels"))
        .agg(avg("recall_at_100").alias("mean_recall_at_100"))
    )
    recall_at_100 = recall_df.collect()[0]["mean_recall_at_100"]

    print(f"{split_name} metrics:")
    print(f"  MAP             = {map_score:.6f}")
    print(f"  NDCG@100        = {ndcg_at_100:.6f}")
    print(f"  precision@100   = {precision_at_100:.6f}")
    print(f"  recall@100      = {recall_at_100:.6f}")

    metrics = spark.createDataFrame(
        [(split_name, map_score, ndcg_at_100, precision_at_100, recall_at_100)],
        ["split", "map", "ndcg_at_100", "precision_at_100", "recall_at_100"]
    )

    output_path = f"{results_path}/{split_name}_metrics.parquet"
    metrics.write.mode("overwrite").parquet(output_path)

    print(f"Saved {split_name} metrics to: {output_path}")


evaluate_split(validation, "validation")
evaluate_split(test, "test")

spark.stop()

