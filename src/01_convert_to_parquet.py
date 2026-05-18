import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType

RAW_INTERACTIONS_PATH = 'hdfs:///user/pw44_nyu_edu/goodreads_interactions.csv'
RAW_USER_MAP_PATH = 'hdfs:///user/pw44_nyu_edu/user_id_map.csv'
RAW_BOOK_MAP_PATH = 'hdfs:///user/pw44_nyu_edu/book_id_map.csv'

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')

args = parser.parse_args()

spark = SparkSession.builder.appName('Goodreads Convert CSV to Parquet').getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

parquet_base = f'{project_base}/parquet'
interactions_out = f'{parquet_base}/interactions.parquet'
user_map_out = f'{parquet_base}/user_id_map.parquet'
book_map_out = f'{parquet_base}/book_id_map.parquet'

interactions_schema = StructType([
    StructField('user_id', IntegerType(), nullable=True),
    StructField('book_id', IntegerType(), nullable=True),
    StructField('is_read', IntegerType(), nullable=True),
    StructField('rating', FloatType(), nullable=True),
    StructField('is_reviewed', IntegerType(), nullable=True)
])

interactions = (
    spark.read
    .option('header', True)
    .schema(interactions_schema)
    .csv(RAW_INTERACTIONS_PATH)
)
user_map = (
    spark.read
    .option('header', True)
    .option('inferSchema', True)
    .csv(RAW_USER_MAP_PATH)
)
book_map = (
    spark.read
    .option('header', True)
    .option('inferSchema', True)
    .csv(RAW_BOOK_MAP_PATH)
)

interactions = interactions.select(
    col('user_id').cast('int').alias('user_id'),
    col('book_id').cast('int').alias('book_id'),
    col('is_read').cast('int').alias('is_read'),
    col('rating').cast('float').alias('rating'),
    col('is_reviewed').cast('int').alias('is_reviewed'))

interactions.write.mode('overwrite').parquet(interactions_out)
user_map.write.mode('overwrite').parquet(user_map_out)
book_map.write.mode('overwrite').parquet(book_map_out)

spark.stop()
