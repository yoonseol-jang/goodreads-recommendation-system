from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, count, row_number, collect_list, collect_set,
    size, array_intersect, avg, lit
)
from pyspark.sql.window import Window


NETID = "dk5406"

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


# Keep more than 100 candidates because we will remove books
# each user has already seen in train.
top_candidates = (
    popular_books
    .limit(500)
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

    # Take top 100 recommendations for each user
    w = Window.partitionBy("user_id").orderBy(col("pop_rank"))

    recs = (
        candidates
        .withColumn("rn", row_number().over(w))
        .where(col("rn") <= 100)
        .groupBy("user_id")
        .agg(collect_list("book_id").alias("rec_books"))
    )

    eval_table = (
        labels
        .join(recs, on="user_id", how="inner")
        .withColumn("hits", size(array_intersect(col("rec_books"), col("label_books"))))
        .withColumn("num_labels", size(col("label_books")))
        .withColumn("precision_at_100", col("hits") / lit(100.0))
        .withColumn("recall_at_100", col("hits") / col("num_labels"))
    )

    print(f"{split_name} evaluation preview:")
    eval_table.select(
        "user_id", "hits", "num_labels", "precision_at_100", "recall_at_100"
    ).show(20)

    metrics = (
        eval_table
        .agg(
            avg("precision_at_100").alias("mean_precision_at_100"),
            avg("recall_at_100").alias("mean_recall_at_100")
        )
    )

    print(f"{split_name} metrics:")
    metrics.show()

    output_path = f"{results_path}/{split_name}_metrics.parquet"
    metrics.write.mode("overwrite").parquet(output_path)

    print(f"Saved {split_name} metrics to: {output_path}")


evaluate_split(validation, "validation")
evaluate_split(test, "test")

spark.stop()
