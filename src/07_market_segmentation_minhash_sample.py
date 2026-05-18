import argparse

from pyspark.ml.feature import CountVectorizer, MinHashLSH
from pyspark.sql import SparkSession
from pyspark.sql.functions import array_intersect, array_union, col, collect_set, size

parser = argparse.ArgumentParser()
parser.add_argument('--netid', default='')
parser.add_argument('--samplePct', dest='sample_pct', type=int, default=3)

args = parser.parse_args()
sample_pct = args.sample_pct
if sample_pct < 1 or sample_pct > 100:
    parser.error('--samplePct must be between 1 and 100')

sample_label = f'{sample_pct}pct'

spark = SparkSession.builder.appName(
    f'Goodreads Market Segmentation MinHash {sample_label}'
).getOrCreate()

if args.netid:
    project_base = f'hdfs:///user/{args.netid}_nyu_edu/goodreads'
else:
    raise ValueError('--netid is required')

input_path = (
    f'{project_base}/samples/interactions_user_sample_{sample_label}.parquet'
)
output_path = (
    f'{project_base}/results/'
    f'market_segmentation_minhash_{sample_label}/top100_pairs.parquet'
)
df = spark.read.parquet(input_path)
read_df = (
    df
    .where(col('is_read') == 1)
    .select('book_id', 'user_id')
    .distinct()
)
book_users = (
    read_df
    .groupBy('book_id')
    .agg(collect_set(col('user_id').cast('string')).alias('user_tokens'))
    .withColumn('reader_count', size(col('user_tokens')))
)
MIN_READERS = 30
book_users = book_users.where(col('reader_count') >= MIN_READERS)
cv = CountVectorizer(
    inputCol='user_tokens',
    outputCol='features',
    binary=True,
    minDF=1.0,
)
cv_model = cv.fit(book_users)
book_vectors = (
    cv_model
    .transform(book_users)
    .select('book_id', 'reader_count', 'user_tokens', 'features')
)
mh = MinHashLSH(
    inputCol='features',
    outputCol='hashes',
    numHashTables=5,
)
mh_model = mh.fit(book_vectors)
DISTANCE_THRESHOLD = 0.8
candidate_pairs = mh_model.approxSimilarityJoin(
    book_vectors,
    book_vectors,
    DISTANCE_THRESHOLD,
    distCol='jaccard_distance',
)
candidate_pairs = candidate_pairs.where(
    col('datasetA.book_id') < col('datasetB.book_id')
)
pairs_scored = (
    candidate_pairs
    .select(
        col('datasetA.book_id').alias('book_id_1'),
        col('datasetB.book_id').alias('book_id_2'),
        col('datasetA.reader_count').alias('reader_count_1'),
        col('datasetB.reader_count').alias('reader_count_2'),
        col('datasetA.user_tokens').alias('users_1'),
        col('datasetB.user_tokens').alias('users_2'),
        col('jaccard_distance'),
    )
    .withColumn(
        'intersection_size',
        size(array_intersect(col('users_1'), col('users_2'))),
    )
    .withColumn('union_size', size(array_union(col('users_1'), col('users_2'))))
    .withColumn('jaccard_similarity', col('intersection_size') / col('union_size'))
    .drop('users_1', 'users_2')
)
top100 = (
    pairs_scored
    .orderBy(
        col('jaccard_similarity').desc(),
        col('intersection_size').desc(),
        col('reader_count_1').desc(),
        col('reader_count_2').desc(),
    )
    .limit(100)
)
top100.write.mode('overwrite').parquet(output_path)

spark.stop()
