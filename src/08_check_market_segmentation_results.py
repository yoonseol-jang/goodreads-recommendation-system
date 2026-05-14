from pyspark.sql import SparkSession



NETID = "dk5406"

spark = (
    SparkSession.builder
    .appName("Check Market Segmentation Results")
    .getOrCreate()
)

path = f"hdfs:///user/{NETID}_nyu_edu/goodreads/results/market_segmentation_minhash_sample/top100_pairs.parquet"

df = spark.read.parquet(path)

print("Top 100 similar book pairs:")
df.show(100, truncate=False)

spark.stop()
