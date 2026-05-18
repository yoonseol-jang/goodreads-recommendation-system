import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=3)
parser.add_argument('--recPool', type=int, default=500)

args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'

spark = SparkSession.builder.appName(
    f'Export Implicit ALS Results {sample_label}'
).getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

results_base = f'{project_base}/results/implicit_als_{sample_label}'
csv_base = f'{project_base}/csv_exports/{sample_label}/implicit_als'

grid = spark.read.parquet(f'{results_base}/validation_grid/')
grid_ordered = grid.orderBy(col('ndcg_at_100').desc()).cache()
grid_ordered.coalesce(1).write.mode('overwrite').csv(
    f'{csv_base}/12_validation_grid',
    header=True,
)

best_row = grid_ordered.first()
best_rank = best_row['rank']
best_reg = best_row['regParam']
best_alpha = best_row['alpha']
combo_tag = f'rank{best_rank}_reg{best_reg}_alpha{best_alpha}_pool{args.recPool}'

exports = [
    (
        f'{results_base}/{combo_tag}/validation_metrics.parquet',
        '12_best_validation_metrics',
    ),
    (f'{results_base}/{combo_tag}/test_metrics.parquet', '12_best_test_metrics'),
]

for parquet_path, csv_name in exports:
    spark.read.parquet(parquet_path).coalesce(1).write.mode('overwrite').csv(
        f'{csv_base}/{csv_name}',
        header=True,
    )

grid_ordered.unpersist()

spark.stop()
