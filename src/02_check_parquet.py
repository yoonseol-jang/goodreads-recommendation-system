import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import sum as spark_sum

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')

args = parser.parse_args()

spark = SparkSession.builder.appName('Goodreads Export Parquet Checks').getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

path = f'{project_base}/parquet/interactions.parquet'
csv_base = f'{project_base}/csv_exports/parquet_checks'

df = spark.read.parquet(path)

row_count = spark.createDataFrame([(df.count(),)], ['total_rows'])
rating_distribution = df.groupBy('rating').count().orderBy('rating')
read_review_summary = df.select(
    spark_sum(col('is_read')).alias('total_read'),
    spark_sum(col('is_reviewed')).alias('total_reviewed'),
)

row_count.coalesce(1).write.mode('overwrite').csv(
    f'{csv_base}/02_row_count',
    header=True,
)
rating_distribution.coalesce(1).write.mode('overwrite').csv(
    f'{csv_base}/02_rating_distribution',
    header=True,
)
read_review_summary.coalesce(1).write.mode('overwrite').csv(
    f'{csv_base}/02_read_review_summary',
    header=True,
)

spark.stop()
