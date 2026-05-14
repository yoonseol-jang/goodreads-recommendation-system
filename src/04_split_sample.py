from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, row_number, rand
from pyspark.sql.window import Window


NETID = "dk5406"

spark = (
    SparkSession.builder
    .appName("Goodreads Split Sample")
    .getOrCreate()
)

input_path = f"hdfs:///user/{NETID}_nyu_edu/goodreads/samples/interactions_user_sample_1pct.parquet"

train_out = f"hdfs:///user/{NETID}_nyu_edu/goodreads/splits/sample_train.parquet"
val_out = f"hdfs:///user/{NETID}_nyu_edu/goodreads/splits/sample_validation.parquet"
test_out = f"hdfs:///user/{NETID}_nyu_edu/goodreads/splits/sample_test.parquet"

df = spark.read.parquet(input_path)

print("Original sample rows:")
print(df.count())

print("Original sample users:")
print(df.select("user_id").distinct().count())

# --------------------------------------------------
# Keep only users with enough interactions.
# We need enough history to split into train/validation/test.
# Here, we keep users with at least 5 interactions.
# --------------------------------------------------

user_counts = df.groupBy("user_id").agg(count("*").alias("n_interactions"))

eligible_users = user_counts.where(col("n_interactions") >= 5).select("user_id")

df = df.join(eligible_users, on="user_id", how="inner")

print("Rows after filtering users with at least 5 interactions:")
print(df.count())

print("Users after filtering:")
print(df.select("user_id").distinct().count())

# --------------------------------------------------
# Randomly order each user's interactions.
# Then split within each user:
# first 60% -> train
# next 20%  -> validation
# last 20%  -> test
# --------------------------------------------------

w = Window.partitionBy("user_id").orderBy(rand(seed=42))

df_ranked = (
    df
    .withColumn("rn", row_number().over(w))
)

user_counts_after = df_ranked.groupBy("user_id").agg(count("*").alias("n"))

df_ranked = df_ranked.join(user_counts_after, on="user_id", how="inner")

train = df_ranked.where(col("rn") <= col("n") * 0.6)
validation = df_ranked.where((col("rn") > col("n") * 0.6) & (col("rn") <= col("n") * 0.8))
test = df_ranked.where(col("rn") > col("n") * 0.8)

# Drop helper columns
train = train.drop("rn", "n")
validation = validation.drop("rn", "n")
test = test.drop("rn", "n")

print("Train rows:")
print(train.count())

print("Validation rows:")
print(validation.count())

print("Test rows:")
print(test.count())

print("Train users:")
print(train.select("user_id").distinct().count())

print("Validation users:")
print(validation.select("user_id").distinct().count())

print("Test users:")
print(test.select("user_id").distinct().count())

print("Writing train/validation/test splits...")

train.write.mode("overwrite").parquet(train_out)
validation.write.mode("overwrite").parquet(val_out)
test.write.mode("overwrite").parquet(test_out)

print(f"Saved train to: {train_out}")
print(f"Saved validation to: {val_out}")
print(f"Saved test to: {test_out}")

spark.stop()
