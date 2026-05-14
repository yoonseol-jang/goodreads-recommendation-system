from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum


NETID = "dk5406"

spark = (
    SparkSession.builder
    .appName("Goodreads Check Parquet")
    .getOrCreate()
)

path = f"hdfs:///user/{NETID}_nyu_edu/goodreads/parquet/interactions.parquet"

df = spark.read.parquet(path)

print("Schema:")
df.printSchema()

print("Sample rows:")
df.show(10)

print("Total rows:")
print(df.count())

print("Rating distribution:")
df.groupBy("rating").count().orderBy("rating").show()

print("Read/review summary:")
df.select(
    spark_sum(col("is_read")).alias("total_read"),
    spark_sum(col("is_reviewed")).alias("total_reviewed")
).show()

spark.stop()
