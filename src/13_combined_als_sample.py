import argparse

from pyspark.ml.recommendation import ALS, ALSModel
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
    lit,
    row_number,
    size,
    sort_array,
    struct,
)
from pyspark.sql.functions import when
from pyspark.sql.window import Window

TOP_K = 100
RELEVANT_RATING = 4
DEFAULT_RANK = 50
DEFAULT_REG_PARAM = 0.1
parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=1)
parser.add_argument('--rank', type=int, default=DEFAULT_RANK)
parser.add_argument('--regParam', type=float, default=DEFAULT_REG_PARAM)
parser.add_argument(
    '--readWeight',
    type=float,
    required=True)
parser.add_argument(
    '--reviewWeight',
    type=float,
    required=True)
parser.add_argument('--maxIter', type=int, default=5)
parser.add_argument('--recPool', type=int, default=250)
parser.add_argument(
    '--loadModel',
    action='store_true')
args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'
rank = args.rank
reg_param = args.regParam
read_weight = args.readWeight
review_weight = args.reviewWeight
max_iter = args.maxIter
rec_pool = args.recPool
spark = SparkSession.builder.appName(
    f'Goodreads Combined ALS {sample_label} '
    f'rank={rank} reg={reg_param} rw={read_weight} rvw={review_weight}'
).getOrCreate()
if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

split_base = f'{project_base}/splits/{sample_label}'
train_path = f'{split_base}/train.parquet'
validation_path = f'{split_base}/validation.parquet'
test_path = f'{split_base}/test.parquet'
results_path = f'{project_base}/results/combined_als_{sample_label}'
checkpoint_path = f'{project_base}/checkpoints/combined_als_{sample_label}'
csv_base = f'{project_base}/csv_exports/{sample_label}'
spark.sparkContext.setCheckpointDir(checkpoint_path)
train = spark.read.parquet(train_path).cache()
validation = spark.read.parquet(validation_path).cache()
test = spark.read.parquet(test_path).cache()
combined_score = (
    when(col('rating') > 0, col('rating'))
    .when(col('is_reviewed') == 1, lit(review_weight))
    .when(col('is_read') == 1, lit(read_weight))
    .otherwise(lit(0.0))
)
train_combined = (
    train
    .withColumn('combined_score', combined_score)
    .where(col('combined_score') > 0)
    .select('user_id', 'book_id', 'combined_score')
    .cache()
)
combo_tag = f'rank{rank}_reg{reg_param}_rw{read_weight}_rvw{review_weight}'
model_path = f'{results_path}/models/{combo_tag}'
if args.loadModel:
    model = ALSModel.load(model_path)
else:
    als = ALS(
        userCol='user_id',
        itemCol='book_id',
        ratingCol='combined_score',
        implicitPrefs=False,
        rank=rank,
        regParam=reg_param,
        maxIter=max_iter,
        checkpointInterval=2,
        coldStartStrategy='drop',
        seed=42,
    )
    model = als.fit(train_combined)
    model.write().overwrite().save(model_path)
seen_train = train.select('user_id', 'book_id').distinct().cache()
train_users = train_combined.select('user_id').distinct().cache()


def labels_for(eval_df, label_kind):
    if label_kind == 'liked':
        condition = col('rating') >= RELEVANT_RATING
    elif label_kind == 'read':
        condition = col('is_read') == 1
    else:
        raise ValueError(f'Unknown label_kind: {label_kind}')
    return (
        eval_df
        .where(condition)
        .groupBy('user_id')
        .agg(collect_set('book_id').alias('label_books'))
    )


def top_k_recs(model, users_df, k=TOP_K, pool=rec_pool):
    raw_recs = model.recommendForUserSubset(users_df, pool)
    flat = (
        raw_recs
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
    return (
        unseen
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


def evaluate_split(eval_df, split_name, label_kind):
    labels = labels_for(eval_df, label_kind).cache()
    users = (
        labels
        .select('user_id')
        .distinct()
        .join(train_users, on='user_id', how='inner')
        .cache()
    )
    recs = top_k_recs(model, users)
    map_score, ndcg, prec, rec = ranking_metrics(recs, labels, k=TOP_K)
    row = spark.createDataFrame(
        [(
            sample_pct,
            split_name,
            label_kind,
            rank,
            reg_param,
            read_weight,
            review_weight,
            max_iter,
            map_score,
            ndcg,
            prec,
            rec,
        )],
        [
            'sample_pct',
            'split',
            'label_kind',
            'rank',
            'regParam',
            'readWeight',
            'reviewWeight',
            'maxIter',
            'map',
            'ndcg_at_100',
            'precision_at_100',
            'recall_at_100',
        ],
    )
    out_path = f'{results_path}/{split_name}_metrics_{label_kind}/{combo_tag}.parquet'
    row.write.mode('overwrite').parquet(out_path)
    row.coalesce(1).write.mode('overwrite').csv(
        f'{csv_base}/13_{split_name}_metrics_{label_kind}_{combo_tag}',
        header=True)
    if split_name == 'validation' and label_kind == 'liked':
        grid_path = f'{results_path}/validation_grid/{combo_tag}.parquet'
        row.write.mode('overwrite').parquet(grid_path)
    labels.unpersist()
    users.unpersist()


evaluate_split(validation, 'validation', 'liked')
evaluate_split(validation, 'validation', 'read')
evaluate_split(test, 'test', 'liked')
evaluate_split(test, 'test', 'read')
seen_train.unpersist()
train_users.unpersist()
train_combined.unpersist()
train.unpersist()
validation.unpersist()
test.unpersist()

spark.stop()
