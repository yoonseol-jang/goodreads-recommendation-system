from pyspark.sql import SparkSession
from pyspark.sql.functions import col, xxhash64, pmod, lit


NETID = "dk5406"

spark = (
    SparkSession.builder
    .appName("Goodreads Make User Sample")
    .getOrCreate()
)

input_path = f"hdfs:///user/{NETID}_nyu_edu/goodreads/parquet/interactions.parquet"
output_path = f"hdfs:///user/{NETID}_nyu_edu/goodreads/samples/interactions_user_sample_1pct.parquet"

interactions = spark.read.parquet(input_path)

# Deterministic user-based sample:
# Keep about 1% of users using a hash of user_id.
sample_users = (
    interactions
    .select("user_id")
    .distinct()
    .where(pmod(xxhash64(col("user_id")), lit(100)) == 0)
)

sample_interactions = interactions.join(sample_users, on="user_id", how="inner")

print("Sample interaction rows:")
print(sample_interactions.count())

print("Sample users:")
print(sample_interactions.select("user_id").distinct().count())

print("Sample books:")
print(sample_interactions.select("book_id").distinct().count())

print("Sample preview:")
sample_interactions.show(10)

sample_interactions.write.mode("overwrite").parquet(output_path)

print(f"Saved sample to: {output_path}")

spark.stop()
