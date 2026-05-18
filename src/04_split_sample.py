import argparse

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, rand, row_number
from pyspark.sql.window import Window

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=3)

args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'

spark = SparkSession.builder.appName(f'Goodreads Split {sample_label} Sample').getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

input_path = (
    f'{project_base}/samples/interactions_user_sample_{sample_label}.parquet'
)
split_base = f'{project_base}/splits/{sample_label}'
train_output_path = f'{split_base}/train.parquet'
validation_output_path = f'{split_base}/validation.parquet'
test_output_path = f'{split_base}/test.parquet'
df = spark.read.parquet(input_path)

user_counts = df.groupBy('user_id').agg(count('*').alias('n_interactions'))
eligible_users = user_counts.where(col('n_interactions') >= 5).select('user_id')
df = df.join(eligible_users, on='user_id', how='inner')

w_order = Window.partitionBy('user_id').orderBy(rand(seed=42))
w_part = Window.partitionBy('user_id')

df_ranked = (
    df
    .withColumn('rn', row_number().over(w_order))
    .withColumn('n', count('*').over(w_part))
    .cache()
)

train = df_ranked.where(col('rn') <= col('n') * 0.6)
validation = df_ranked.where((col('rn') > col('n') * 0.6) & (col('rn') <= col('n') * 0.8))
test = df_ranked.where(col('rn') > col('n') * 0.8)

train = train.drop('rn', 'n')
validation = validation.drop('rn', 'n')
test = test.drop('rn', 'n')

train.write.mode('overwrite').parquet(train_output_path)
validation.write.mode('overwrite').parquet(validation_output_path)
test.write.mode('overwrite').parquet(test_output_path)

spark.stop()
