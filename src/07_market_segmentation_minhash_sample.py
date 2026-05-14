from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, collect_set, size, array_intersect, array_union,
    lit
)
from pyspark.ml.feature import CountVectorizer, MinHashLSH



NETID = "dk5406"

spark = (
    SparkSession.builder
    .appName("Goodreads Market Segmentation MinHash Sample")
    .getOrCreate()
)

base = f"hdfs:///user/{NETID}_nyu_edu/goodreads"

# Use sample data first
input_path = f"{base}/samples/interactions_user_sample_1pct.parquet"

output_path = f"{base}/results/market_segmentation_minhash_sample/top100_pairs.parquet"

df = spark.read.parquet(input_path)

# --------------------------------------------------
# 1. Keep only read interactions
# Market position = set of users who read the book
# --------------------------------------------------

read_df = (
    df
    .where(col("is_read") == 1)
    .select("book_id", "user_id")
    .distinct()
)

print("Read interaction rows:")
print(read_df.count())

# --------------------------------------------------
# 2. Build user set for each book.
# Each book becomes a set of reader user_ids.
# --------------------------------------------------

book_users = (
    read_df
    .groupBy("book_id")
    .agg(collect_set(col("user_id").cast("string")).alias("user_tokens"))
    .withColumn("reader_count", size(col("user_tokens")))
)

# For sample data, filter out books with too few readers.

MIN_READERS = 10

book_users = book_users.where(col("reader_count") >= MIN_READERS)

print("Books after reader-count filtering:")
print(book_users.count())

print("Book user-set preview:")
book_users.show(10, truncate=False)

# --------------------------------------------------
# 3. Convert user sets into sparse vectors.
# CountVectorizer with binary=True records whether each user is in the set.
# --------------------------------------------------

cv = CountVectorizer(
    inputCol="user_tokens",
    outputCol="features",
    binary=True,
    minDF=1.0
)

cv_model = cv.fit(book_users)
book_vectors = cv_model.transform(book_users).select(
    "book_id",
    "reader_count",
    "user_tokens",
    "features"
)

print("Book vectors preview:")
book_vectors.show(5, truncate=False)

# --------------------------------------------------
# 4. Use MinHashLSH to find similar book pairs.
# distCol = Jaccard distance = 1 - Jaccard similarity.
# --------------------------------------------------

mh = MinHashLSH(
    inputCol="features",
    outputCol="hashes",
    numHashTables=5
)

mh_model = mh.fit(book_vectors)

# Distance threshold:
DISTANCE_THRESHOLD = 0.8

candidate_pairs = mh_model.approxSimilarityJoin(
    book_vectors,
    book_vectors,
    DISTANCE_THRESHOLD,
    distCol="jaccard_distance"
)

# Remove self-pairs and duplicated symmetric pairs.
candidate_pairs = (
    candidate_pairs
    .where(col("datasetA.book_id") < col("datasetB.book_id"))
)

print("Candidate pair count:")
print(candidate_pairs.count())

# --------------------------------------------------
# 5. Compute exact Jaccard similarity for candidate pairs.
# --------------------------------------------------

pairs_scored = (
    candidate_pairs
    .select(
        col("datasetA.book_id").alias("book_id_1"),
        col("datasetB.book_id").alias("book_id_2"),
        col("datasetA.reader_count").alias("reader_count_1"),
        col("datasetB.reader_count").alias("reader_count_2"),
        col("datasetA.user_tokens").alias("users_1"),
        col("datasetB.user_tokens").alias("users_2"),
        col("jaccard_distance")
    )
    .withColumn("intersection_size", size(array_intersect(col("users_1"), col("users_2"))))
    .withColumn("union_size", size(array_union(col("users_1"), col("users_2"))))
    .withColumn("jaccard_similarity", col("intersection_size") / col("union_size"))
    .drop("users_1", "users_2")
)

top100 = (
    pairs_scored
    .orderBy(
        col("jaccard_similarity").desc(),
        col("intersection_size").desc(),
        col("reader_count_1").desc(),
        col("reader_count_2").desc()
    )
    .limit(100)
)

print("Top 100 similar book pairs:")
top100.show(100, truncate=False)

top100.write.mode("overwrite").parquet(output_path)

print(f"Saved top 100 pairs to: {output_path}")

spark.stop()
