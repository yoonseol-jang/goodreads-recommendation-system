from pyspark.sql import SparkSession


NETID = "dk5406"

spark = (
    SparkSession.builder
    .appName("Check Popularity Baseline Results")
    .getOrCreate()
)

base = f"hdfs:///user/{NETID}_nyu_edu/goodreads/results/popularity_baseline_sample"

validation_path = f"{base}/validation_metrics.parquet"
test_path = f"{base}/test_metrics.parquet"

print("Validation metrics:")
validation = spark.read.parquet(validation_path)
validation.show(truncate=False)

print("Test metrics:")
test = spark.read.parquet(test_path)
test.show(truncate=False)

spark.stop()
