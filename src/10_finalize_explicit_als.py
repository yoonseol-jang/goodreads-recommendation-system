import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Aggregate explicit ALS validation rows and evaluate the winner on test.'
        )
    )
    parser.add_argument('--netid', default='')
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
    return parser.parse_args()


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
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

args = parse_args()
SAMPLE_TAG = args.sampleTag
SPLIT_PREFIX = args.splitPrefix
TOP_K = 100
RELEVANT_RATING = 4

spark = SparkSession.builder.appName(
    'Goodreads Explicit ALS Finalize').getOrCreate()
if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

if args.splitLayout == 'nested':
    split_base = f'{project_base}/splits/{SAMPLE_TAG}'
    train_path = f'{split_base}/train.parquet'
    validation_path = f'{split_base}/validation.parquet'
    test_path = f'{split_base}/test.parquet'
else:
    split_base = f'{project_base}/splits'
    train_path = f'{split_base}/{SPLIT_PREFIX}_train.parquet'
    validation_path = f'{split_base}/{SPLIT_PREFIX}_validation.parquet'
    test_path = f'{split_base}/{SPLIT_PREFIX}_test.parquet'

results_path = f'{project_base}/results/explicit_als_sample_{SAMPLE_TAG}_recpool'


def hdfs_exists(path):
    jvm = spark.sparkContext._jvm
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    fs = jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    return fs.exists(jvm.org.apache.hadoop.fs.Path(path))


def load_best_model(rank, reg_param, rec_pool):
    tags = []
    for tag in (f'rank={rank}_reg={reg_param}_pool={rec_pool}',
                f'rank={rank}_reg={reg_param:g}_pool={rec_pool}'):
        if tag not in tags:
            tags.append(tag)
    tried_paths = [f'{results_path}/models/{tag}' for tag in tags]
    for path in tried_paths:
        if hdfs_exists(path):
            return ALSModel.load(path)
    raise FileNotFoundError(
        'Could not find saved ALS model for best validation row. Tried: ' +
        ', '.join(tried_paths))


grid_df = spark.read.option(
    'recursiveFileLookup',
    'true').parquet(f'{results_path}/validation_grid/').cache()
grid_df.write.mode('overwrite').parquet(
    f'{results_path}/validation_grid.parquet')
best_row = grid_df.orderBy(col('ndcg_at_100').desc()).first()
best_rank = int(best_row['rank'])
best_reg_param = float(best_row['regParam'])
best_rec_pool = int(best_row['recPool'])
best_max_iter = int(best_row['maxIter'])
validation_map = best_row['map']
validation_ndcg_at_100 = best_row['ndcg_at_100']
validation_precision_at_100 = best_row['precision_at_100']
validation_recall_at_100 = best_row['recall_at_100']
train_all = spark.read.parquet(train_path).cache()
train = train_all.where(col('rating') > 0).cache()
validation = spark.read.parquet(validation_path).cache()
test = spark.read.parquet(test_path).cache()
seen_train = train_all.select('user_id', 'book_id').distinct().cache()
train_users = train.select('user_id').distinct().cache()
best_model = load_best_model(best_rank, best_reg_param, best_rec_pool)


def top_k_recs(model, users_df, pool, k=TOP_K):
    raw = model.recommendForUserSubset(users_df, pool)
    flat = raw.select('user_id',
                      explode('recommendations').alias('rec')).select(
                          'user_id',
                          col('rec.book_id').alias('book_id'),
                          col('rec.rating').alias('score'))
    unseen = flat.join(broadcast(seen_train),
                       on=['user_id', 'book_id'],
                       how='left_anti')
    w = Window.partitionBy('user_id').orderBy(col('score').desc())
    top = unseen.withColumn('rn', row_number().over(w)).where(col('rn') <= k)
    return top.groupBy('user_id').agg(
        sort_array(collect_list(struct(
            'rn', 'book_id'))).alias('ranked')).withColumn(
                'rec_books', expr('transform(ranked, x -> x.book_id)')).select(
                    'user_id', 'rec_books')


def ranking_metrics(recs_df, eval_df, k=TOP_K):
    labels = eval_df.where(
        col('rating') >= RELEVANT_RATING).groupBy('user_id').agg(
            collect_set('book_id').alias('label_books'))
    joined = labels.join(recs_df, on='user_id', how='inner').cache()
    pred_and_labels = joined.select('rec_books', 'label_books').rdd.map(
        lambda r: (list(r['rec_books']), list(r['label_books'])))
    rm = RankingMetrics(pred_and_labels)
    map_score = rm.meanAveragePrecision
    ndcg = rm.ndcgAt(k)
    precision = rm.precisionAt(k)
    recall_row = joined.withColumn(
        'hits', size(array_intersect(
            col('rec_books'), col('label_books')))).withColumn(
                'num_labels', size(col('label_books'))).withColumn(
                    'recall',
                    col('hits') / col('num_labels')).agg(
                        avg('recall').alias('mean_recall')).collect()[0]
    recall = recall_row['mean_recall']
    joined.unpersist()
    return (map_score, ndcg, precision, recall)


def evaluate_users(eval_df):
    return eval_df.where(
        col('rating') >= RELEVANT_RATING).select('user_id').distinct().join(
            train_users, on='user_id', how='inner')


rmse_eval = RegressionEvaluator(metricName='rmse',
                                labelCol='rating',
                                predictionCol='prediction')
rated_val = validation.where(col('rating') > 0)
val_pred = best_model.transform(rated_val).where(col('prediction').isNotNull())
validation_rmse = rmse_eval.evaluate(val_pred)
val_metrics = spark.createDataFrame(
    [(
        'validation',
        best_rank,
        best_reg_param,
        best_rec_pool,
        best_max_iter,
        validation_map,
        validation_ndcg_at_100,
        validation_precision_at_100,
        validation_recall_at_100,
        validation_rmse,
    )], [
          'split', 'rank', 'regParam', 'recPool', 'maxIter', 'map',
          'ndcg_at_100', 'precision_at_100', 'recall_at_100', 'rmse'
      ])
val_metrics.write.mode('overwrite').parquet(
    f'{results_path}/validation_metrics.parquet')
test_users = evaluate_users(test).cache()
test_recs = top_k_recs(best_model, test_users, pool=best_rec_pool)
test_map, test_ndcg_at_100, test_precision_at_100, test_recall_at_100 = (
    ranking_metrics(test_recs, test)
)
rated_test = test.where(col('rating') > 0)
test_pred = best_model.transform(rated_test).where(
    col('prediction').isNotNull())
test_rmse = rmse_eval.evaluate(test_pred)
test_metrics = spark.createDataFrame(
    [(
        'test',
        best_rank,
        best_reg_param,
        best_rec_pool,
        best_max_iter,
        test_map,
        test_ndcg_at_100,
        test_precision_at_100,
        test_recall_at_100,
        test_rmse,
    )], [
          'split', 'rank', 'regParam', 'recPool', 'maxIter', 'map',
          'ndcg_at_100', 'precision_at_100', 'recall_at_100', 'rmse'
      ])
test_metrics.write.mode('overwrite').parquet(f'{results_path}/test_metrics.parquet')

spark.stop()
