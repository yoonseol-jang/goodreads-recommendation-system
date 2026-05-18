import argparse

from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array_intersect,
    avg,
    col,
    collect_list,
    collect_set,
    count,
    expr,
    row_number,
    size,
    sort_array,
    struct,
)
from pyspark.sql.window import Window

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=3)

args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'

spark = SparkSession.builder.appName(
    f'Goodreads Popularity Baseline {sample_label}'
).getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

split_base = f'{project_base}/splits/{sample_label}'
train_path = f'{split_base}/train.parquet'
validation_path = f'{split_base}/validation.parquet'
test_path = f'{split_base}/test.parquet'
results_path = f'{project_base}/results/popularity_baseline_{sample_label}'
train = spark.read.parquet(train_path).cache()
validation = spark.read.parquet(validation_path).cache()
test = spark.read.parquet(test_path).cache()

train_users = train.select('user_id').distinct().cache()
rated_train_users = (
    train
    .where(col('rating') > 0)
    .select('user_id')
    .distinct()
    .cache()
)
seen_train = train.select('user_id', 'book_id').distinct().cache()

popular_books = (
    train
    .where(col('is_read') == 1)
    .groupBy('book_id')
    .agg(count('*').alias('read_count'))
    .orderBy(col('read_count').desc())
)

top_candidates = (
    popular_books
    .limit(1000)
    .withColumn(
        'pop_rank',
        row_number().over(Window.orderBy(col('read_count').desc())),
    )
    .select('book_id', 'read_count', 'pop_rank')
)

w_rank = Window.partitionBy('user_id').orderBy(col('pop_rank'))
all_recs = (
    train_users
    .crossJoin(top_candidates)
    .join(seen_train, on=['user_id', 'book_id'], how='left_anti')
    .withColumn('rn', row_number().over(w_rank))
    .where(col('rn') <= 100)
    .groupBy('user_id')
    .agg(sort_array(collect_list(struct('rn', 'book_id'))).alias('ranked'))
    .withColumn('rec_books', expr('transform(ranked, x -> x.book_id)'))
    .select('user_id', 'rec_books')
    .cache()
)


def evaluate_split(
    eval_df,
    split_name,
    label_name,
    label_filter,
    user_filter_df,
    output_suffix,
):
    labels = (
        eval_df
        .where(label_filter)
        .groupBy('user_id')
        .agg(collect_set('book_id').alias('label_books'))
    )

    eligible = (
        labels
        .select('user_id')
        .join(user_filter_df, on='user_id', how='inner')
    )

    eval_table = (
        labels
        .join(eligible, on='user_id', how='inner')
        .join(all_recs, on='user_id', how='inner')
    )

    pred_and_labels = (
        eval_table
        .select('rec_books', 'label_books')
        .rdd
        .map(lambda r: (list(r['rec_books']), list(r['label_books'])))
    )

    rm = RankingMetrics(pred_and_labels)
    map_score = rm.meanAveragePrecision
    ndcg_at_100 = rm.ndcgAt(100)
    precision_at_100 = rm.precisionAt(100)

    recall_df = (
        eval_table
        .withColumn(
            'hits',
            size(array_intersect(col('rec_books'), col('label_books'))),
        )
        .withColumn('num_labels', size(col('label_books')))
        .withColumn('recall_at_100', col('hits') / col('num_labels'))
        .agg(avg('recall_at_100').alias('mean_recall_at_100'))
    )
    recall_at_100 = recall_df.collect()[0]['mean_recall_at_100']

    metrics = spark.createDataFrame(
        [(
            split_name,
            label_name,
            map_score,
            ndcg_at_100,
            precision_at_100,
            recall_at_100,
        )],
        [
            'split',
            'label_definition',
            'map',
            'ndcg_at_100',
            'precision_at_100',
            'recall_at_100',
        ],
    )
    output_path = f'{results_path}/{split_name}{output_suffix}_metrics.parquet'
    metrics.write.mode('overwrite').parquet(output_path)


evaluate_split(
    validation,
    'validation',
    'is_read=1',
    col('is_read') == 1,
    train_users,
    '',
)
evaluate_split(test, 'test', 'is_read=1', col('is_read') == 1, train_users, '')
evaluate_split(
    validation,
    'validation',
    'rating>=4',
    col('rating') >= 4,
    rated_train_users,
    '_rating4',
)
evaluate_split(
    test,
    'test',
    'rating>=4',
    col('rating') >= 4,
    rated_train_users,
    '_rating4',
)

spark.stop()
