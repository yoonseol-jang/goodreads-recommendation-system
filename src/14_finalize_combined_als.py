import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=1)

args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'

spark = SparkSession.builder.appName(
    f'Finalize Combined ALS {sample_label}'
).getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

results_base = f'{project_base}/results/combined_als_{sample_label}'
csv_base = f'{project_base}/csv_exports/{sample_label}'


def path_exists(path):
    jvm = spark.sparkContext._jvm
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    return jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf).exists(
        jvm.org.apache.hadoop.fs.Path(path))


def show_and_export(path, csv_name):
    if not path_exists(path):
        return None
    df = spark.read.parquet(path)
    ordered = df.orderBy(col('ndcg_at_100').desc()).cache()
    ordered.coalesce(1).write.mode('overwrite').csv(f'{csv_base}/{csv_name}',
                                                    header=True)
    return ordered


grid_path = f'{results_base}/validation_grid/'
grid_ordered = show_and_export(grid_path, '14_validation_grid')
if grid_ordered is None:
    spark.stop()
    raise SystemExit(1)
for label_kind in ('liked', 'read'):
    for split_name in ('validation', 'test'):
        show_and_export(
            f'{results_base}/{split_name}_metrics_{label_kind}/',
            f'14_{split_name}_metrics_{label_kind}',
        )
grid_ordered.unpersist()

spark.stop()
