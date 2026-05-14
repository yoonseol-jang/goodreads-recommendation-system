from pyspark.sql import SparkSession
from pyspark.sql.functions import col


NETID = "dk5406"


spark = (
    SparkSession.builder
    .appName("Goodreads Convert CSV to Parquet")
    .getOrCreate()
)


raw_interactions_path = "hdfs:///user/pw44_nyu_edu/goodreads_interactions.csv"
raw_user_map_path = "hdfs:///user/pw44_nyu_edu/user_id_map.csv"
raw_book_map_path = "hdfs:///user/pw44_nyu_edu/book_id_map.csv"

output_base = f"hdfs:///user/{NETID}_nyu_edu/goodreads/parquet"

interactions_out = f"{output_base}/interactions.parquet"
user_map_out = f"{output_base}/user_id_map.parquet"
book_map_out = f"{output_base}/book_id_map.parquet"


interactions = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(raw_interactions_path)
)

user_map = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(raw_user_map_path)
)

book_map = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(raw_book_map_path)
)


print("Interactions columns:")
print(interactions.columns)

print("User map columns:")
print(user_map.columns)

print("Book map columns:")
print(book_map.columns)

interactions = (
    interactions
    .select(
        col("user_id").cast("int").alias("user_id"),
        col("book_id").cast("int").alias("book_id"),
        col("is_read").cast("int").alias("is_read"),
        col("rating").cast("float").alias("rating"),
        col("is_reviewed").cast("int").alias("is_reviewed")
    )
)


print("Interactions schema:")
interactions.printSchema()

print("Sample interactions:")
interactions.show(5)


print("Writing interactions parquet...")
interactions.write.mode("overwrite").parquet(interactions_out)

print("Writing user map parquet...")
user_map.write.mode("overwrite").parquet(user_map_out)

print("Writing book map parquet...")
book_map.write.mode("overwrite").parquet(book_map_out)


print("Finished writing Parquet files.")
print(f"Interactions saved to: {interactions_out}")
print(f"User map saved to: {user_map_out}")
print(f"Book map saved to: {book_map_out}")

spark.stop()
