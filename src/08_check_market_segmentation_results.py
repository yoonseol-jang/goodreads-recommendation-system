import argparse

from pyspark.sql import SparkSession

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=3)

args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'

spark = SparkSession.builder.appName(
    f'Export Market Segmentation Results {sample_label}'
).getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

parquet_path = (
    f'{project_base}/results/'
    f'market_segmentation_minhash_{sample_label}/top100_pairs.parquet'
)
csv_path = f'{project_base}/csv_exports/{sample_label}/market_segmentation/08_top100_pairs'

spark.read.parquet(parquet_path).coalesce(1).write.mode('overwrite').csv(
    csv_path,
    header=True,
)

spark.stop()
