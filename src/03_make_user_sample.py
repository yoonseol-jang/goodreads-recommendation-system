import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, pmod, xxhash64

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=3)

args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'

spark = SparkSession.builder.appName(
    f'Goodreads Make {sample_label} User Sample'
).getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

input_path = f'{project_base}/parquet/interactions.parquet'
output_path = (
    f'{project_base}/samples/interactions_user_sample_{sample_label}.parquet'
)

interactions = spark.read.parquet(input_path)
sample_interactions = interactions.where(pmod(xxhash64(col('user_id')), lit(100)) < sample_pct)

sample_interactions.write.mode('overwrite').parquet(output_path)

spark.stop()
