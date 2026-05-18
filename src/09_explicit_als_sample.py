import argparse

from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array_intersect,
    avg,
    broadcast,
    col,
    collect_list,
    collect_set,
    explode,
    expr,
    row_number,
    size,
    sort_array,
    struct,
)
from pyspark.sql.window import Window


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train one explicit-feedback ALS configuration on a sampled split.',
    )
    parser.add_argument('--netid', default='')
    parser.add_argument(
        '--rank',
        type=int,
        required=True,
    )
    parser.add_argument(
        '--regParam',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--recPool',
        type=int,
        default=None,
    )
    parser.add_argument(
        '--sampleTag',
        default='3pct',
    )
    parser.add_argument(
        '--splitPrefix',
        default='sample',
    )
    parser.add_argument(
        '--splitLayout',
        choices=('flat', 'nested'),
        default='nested',
    )
    parser.add_argument('--maxIter', type=int, default=5)
    args = parser.parse_args()

    rank = args.rank
    reg_param = args.regParam
    rec_pool = args.recPool

    if rec_pool is None:
        rec_pool = 250

    return args, rank, reg_param, rec_pool


args, rank, reg_param, rec_pool = parse_args()
SAMPLE_TAG = args.sampleTag
SPLIT_PREFIX = args.splitPrefix
SEED = 42
TOP_K = 100
max_iter = args.maxIter
spark = SparkSession.builder.appName(
    f'Goodreads Explicit ALS rank={rank} reg={reg_param} pool={rec_pool}'
).getOrCreate()
if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

spark.sparkContext.setCheckpointDir(f'{project_base}/checkpoints')
if args.splitLayout == 'nested':
    split_base = f'{project_base}/splits/{SAMPLE_TAG}'
    train_path = f'{split_base}/train.parquet'
    validation_path = f'{split_base}/validation.parquet'
else:
    split_base = f'{project_base}/splits'
    train_path = f'{split_base}/{SPLIT_PREFIX}_train.parquet'
    validation_path = f'{split_base}/{SPLIT_PREFIX}_validation.parquet'

results_path = f'{project_base}/results/explicit_als_sample_{SAMPLE_TAG}_recpool'
combo_tag = f'rank={rank}_reg={reg_param}_pool={rec_pool}'
model_out = f'{results_path}/models/{combo_tag}'
grid_row_out = f'{results_path}/validation_grid/{combo_tag}.parquet'
train_all = spark.read.parquet(train_path).cache()
train = train_all.where(col('rating') > 0).cache()
validation = spark.read.parquet(validation_path).cache()
seen_train = train_all.select('user_id', 'book_id').distinct().cache()
train_users = train.select('user_id').distinct().cache()
RELEVANT_RATING = 4


def top_k_recs(model, users_df, k=TOP_K, pool=rec_pool):
    raw = model.recommendForUserSubset(users_df, pool)
    flat = (
        raw
        .select('user_id', explode('recommendations').alias('rec'))
        .select(
            'user_id',
            col('rec.book_id').alias('book_id'),
            col('rec.rating').alias('score'),
        )
    )
    unseen = flat.join(
        broadcast(seen_train),
        on=['user_id', 'book_id'],
        how='left_anti',
    )
    w = Window.partitionBy('user_id').orderBy(col('score').desc())
    top = unseen.withColumn('rn', row_number().over(w)).where(col('rn') <= k)
    return (
        top
        .groupBy('user_id')
        .agg(sort_array(collect_list(struct('rn', 'book_id'))).alias('ranked'))
        .withColumn('rec_books', expr('transform(ranked, x -> x.book_id)'))
        .select('user_id', 'rec_books')
    )


def ranking_metrics(recs_df, eval_df, k=TOP_K):
    labels = (
        eval_df
        .where(col('rating') >= RELEVANT_RATING)
        .groupBy('user_id')
        .agg(collect_set('book_id').alias('label_books'))
    )
    joined = labels.join(recs_df, on='user_id', how='inner').cache()
    pred_and_labels = (
        joined
        .select('rec_books', 'label_books')
        .rdd
        .map(lambda r: (list(r['rec_books']), list(r['label_books'])))
    )
    rm = RankingMetrics(pred_and_labels)
    map_score = rm.meanAveragePrecision
    ndcg_at_k = rm.ndcgAt(k)
    precision_at_k = rm.precisionAt(k)
    recall_row = (
        joined
        .withColumn(
            'hits',
            size(array_intersect(col('rec_books'), col('label_books'))),
        )
        .withColumn('num_labels', size(col('label_books')))
        .withColumn('recall', col('hits') / col('num_labels'))
        .agg(avg('recall').alias('mean_recall'))
        .collect()[0]
    )
    recall_at_k = recall_row['mean_recall']
    joined.unpersist()
    return (map_score, ndcg_at_k, precision_at_k, recall_at_k)


def evaluate_users(eval_df):
    return (
        eval_df
        .where(col('rating') >= RELEVANT_RATING)
        .select('user_id')
        .distinct()
        .join(train_users, on='user_id', how='inner')
    )


val_users = evaluate_users(validation).cache()
als = ALS(
    rank=rank,
    regParam=reg_param,
    maxIter=max_iter,
    userCol='user_id',
    itemCol='book_id',
    ratingCol='rating',
    coldStartStrategy='drop',
    nonnegative=False,
    seed=SEED,
    checkpointInterval=2,
)
model = als.fit(train)
recs = top_k_recs(model, val_users)
map_score, ndcg_at_100, precision_at_100, recall_at_100 = ranking_metrics(
    recs,
    validation,
)
model.write().overwrite().save(model_out)
grid_row = spark.createDataFrame(
    [(
        rank,
        reg_param,
        rec_pool,
        max_iter,
        map_score,
        ndcg_at_100,
        precision_at_100,
        recall_at_100,
    )],
    [
        'rank',
        'regParam',
        'recPool',
        'maxIter',
        'map',
        'ndcg_at_100',
        'precision_at_100',
        'recall_at_100',
    ],
)
grid_row.write.mode('overwrite').parquet(grid_row_out)

spark.stop()
