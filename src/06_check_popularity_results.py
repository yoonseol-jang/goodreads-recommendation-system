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
    f'Export Popularity Baseline Results {sample_label}'
).getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

results_base = f'{project_base}/results/popularity_baseline_{sample_label}'
csv_base = f'{project_base}/csv_exports/{sample_label}/popularity_baseline'

exports = [
    (f'{results_base}/validation_metrics.parquet', '06_validation_metrics'),
    (f'{results_base}/test_metrics.parquet', '06_test_metrics'),
    (
        f'{results_base}/validation_rating4_metrics.parquet',
        '06_validation_rating4_metrics',
    ),
    (f'{results_base}/test_rating4_metrics.parquet', '06_test_rating4_metrics'),
]

for parquet_path, csv_name in exports:
    spark.read.parquet(parquet_path).coalesce(1).write.mode('overwrite').csv(
        f'{csv_base}/{csv_name}',
        header=True,
    )

spark.stop()
