import argparse

from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    array_intersect,
    avg,
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

TOP_K = 100
parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=3)
parser.add_argument('--rank', type=int, default=100)
parser.add_argument('--regParam', type=float, default=0.1)
parser.add_argument('--alpha', type=float, default=15.0)
parser.add_argument('--maxIter', type=int, default=5)
parser.add_argument('--recPool', type=int, default=500)
args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'
rank = args.rank
reg_param = args.regParam
alpha = args.alpha
max_iter = args.maxIter
rec_pool = args.recPool
combo_tag = f'rank{rank}_reg{reg_param}_alpha{alpha}_pool{rec_pool}'
spark = SparkSession.builder.appName(
    f'Goodreads Implicit ALS {sample_label} '
    f'rank={rank} reg={reg_param} alpha={alpha}'
).getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

split_base = f'{project_base}/splits/{sample_label}'
train_path = f'{split_base}/train.parquet'
validation_path = f'{split_base}/validation.parquet'
test_path = f'{split_base}/test.parquet'
results_path = f'{project_base}/results/implicit_als_{sample_label}'
checkpoint_path = f'{project_base}/checkpoints/implicit_als_{sample_label}'

spark.sparkContext.setCheckpointDir(checkpoint_path)
train = spark.read.parquet(train_path).cache()
validation = spark.read.parquet(validation_path).cache()
test = spark.read.parquet(test_path).cache()

train_implicit = (
    train
    .withColumn(
        'implicit_score',
        col('is_read').cast('float') + 2.0 * col('is_reviewed').cast('float'),
    )
    .where(col('implicit_score') > 0)
    .select('user_id', 'book_id', 'implicit_score')
    .cache()
)
als = ALS(
    userCol='user_id',
    itemCol='book_id',
    ratingCol='implicit_score',
    implicitPrefs=True,
    rank=rank,
    regParam=reg_param,
    alpha=alpha,
    maxIter=max_iter,
    checkpointInterval=2,
    coldStartStrategy='drop',
    nonnegative=True,
    seed=42,
)

model = als.fit(train_implicit)
model_path = f'{results_path}/models/{combo_tag}'
model.write().overwrite().save(model_path)
seen_train = train.select('user_id', 'book_id').distinct().cache()


def make_labels(eval_df):
    return (
        eval_df
        .where(col('is_read') == 1)
        .groupBy('user_id')
        .agg(collect_set('book_id').alias('label_books'))
    )


def top_k_recs(model, users_df, k=TOP_K, pool=rec_pool):
    raw_recs = model.recommendForUserSubset(users_df, pool)
    recs_exploded = (
        raw_recs
        .select('user_id', explode('recommendations').alias('rec'))
        .select(
            'user_id',
            col('rec.book_id').alias('book_id'),
            col('rec.rating').alias('score'),
        )
    )
    recs_filtered = recs_exploded.join(
        seen_train,
        on=['user_id', 'book_id'],
        how='left_anti',
    )
    w = Window.partitionBy('user_id').orderBy(col('score').desc())
    return (
        recs_filtered
        .withColumn('rn', row_number().over(w))
        .where(col('rn') <= k)
        .groupBy('user_id')
        .agg(sort_array(collect_list(struct('rn', 'book_id'))).alias('ranked'))
        .withColumn('rec_books', expr('transform(ranked, x -> x.book_id)'))
        .select('user_id', 'rec_books')
    )


def ranking_metrics(recs_df, labels_df, k=TOP_K):
    eval_table = labels_df.join(recs_df, on='user_id', how='inner').cache()
    pred_and_labels = (
        eval_table
        .select('rec_books', 'label_books')
        .rdd
        .map(lambda r: (list(r['rec_books']), list(r['label_books'])))
    )
    rm = RankingMetrics(pred_and_labels)
    map_score = rm.meanAveragePrecision
    ndcg_at_k = rm.ndcgAt(k)
    precision_at_k = rm.precisionAt(k)
    recall_row = (
        eval_table
        .withColumn(
            'hits',
            size(array_intersect(col('rec_books'), col('label_books'))),
        )
        .withColumn('num_labels', size(col('label_books')))
        .withColumn('recall_at_k', col('hits') / col('num_labels'))
        .agg(avg('recall_at_k').alias('mean_recall_at_k'))
        .collect()[0]
    )
    recall_at_k = recall_row['mean_recall_at_k']
    eval_table.unpersist()
    return (map_score, ndcg_at_k, precision_at_k, recall_at_k)


def evaluate_split(eval_df, split_name):
    labels = make_labels(eval_df).cache()
    users = (
        labels
        .select('user_id')
        .distinct()
        .join(
            train_implicit.select('user_id').distinct(),
            on='user_id',
            how='inner',
        )
        .cache()
    )
    recs = top_k_recs(model, users)
    map_score, ndcg_at_100, precision_at_100, recall_at_100 = ranking_metrics(
        recs, labels, k=TOP_K)
    metrics = spark.createDataFrame(
        [(
            split_name,
            sample_label,
            rank,
            reg_param,
            alpha,
            max_iter,
            map_score,
            ndcg_at_100,
            precision_at_100,
            recall_at_100,
        )],
        [
            'split',
            'sample',
            'rank',
            'regParam',
            'alpha',
            'maxIter',
            'map',
            'ndcg_at_100',
            'precision_at_100',
            'recall_at_100',
        ],
    )
    output_path = f'{results_path}/{combo_tag}/{split_name}_metrics.parquet'
    metrics.write.mode('overwrite').parquet(output_path)
    if split_name == 'validation':
        grid_path = f'{results_path}/validation_grid/{combo_tag}.parquet'
        metrics.write.mode('overwrite').parquet(grid_path)
    labels.unpersist()
    users.unpersist()


evaluate_split(validation, 'validation')
evaluate_split(test, 'test')
seen_train.unpersist()
train_implicit.unpersist()
train.unpersist()
validation.unpersist()
test.unpersist()

spark.stop()
