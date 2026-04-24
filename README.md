# DSGA1004 - BIG DATA
## Capstone project: Goodreads Recommendation and Market Segmentation

# Overview

In the capstone project, you will apply the tools you have learned in this class to solve a realistic, large-scale applied problem. Specifically, you will use the Goodreads dataset to build and evaluate book recommendation systems, while also using scalable similarity methods to identify books that occupy similar positions in the reading market.

The project is designed to integrate several core themes from the course: distributed storage, large-scale data preprocessing, similarity search, collaborative filtering, recommender-system evaluation, and the practical challenges of working with large user-item interaction data.

You are encouraged to work in groups of up to 4 students.

## The data set

In this project, we will use the Goodreads dataset assembled by:

Mengting Wan and Julian McAuley. “Item Recommendation on Monotonic Behavior Chains.” RecSys 2018.

The core interaction data consists of user-book interactions. The main file has the following column structure:

user_id, book_id, is_read, rating, is_reviewed

The dataset contains approximately 876K users, 2.4M books, and 223M interactions, with three main files: goodreads_interactions.csv, user_id_map.csv, and book_id_map.csv

These shared files are available in Dataproc's HDFS at:

/user/pw44_nyu_edu/user_id_map.csv

/user/pw44_nyu_edu/book_id_map.csv

/user/pw44_nyu_edu/goodreads_interactions.csv


You may read these files directly from HDFS. Do not repeatedly copy or reload the raw CSV unnecessarily. As an early preprocessing step, convert the raw CSV to Parquet in your own HDFS directory and use the Parquet version for downstream work.

The fields can be interpreted as follows:

rating is the explicit preference signal: the numerical rating the user gave the book.
is_read indicates whether the user marked the book as read. This is an implicit consumption signal, not necessarily a sign that the user liked the book.
is_reviewed indicates whether the user wrote a review. This is an engagement signal, not necessarily a positive preference signal.

In other words:

rating       = explicit evaluation

is_read      = implicit consumption

is_reviewed  = implicit engagement



## Preprocessing and data splitting
Before beginning the recommender-system components, you will need to preprocess the interaction data and partition it into training, validation, and test sets.

We recommend writing scripts to do this in advance and saving the resulting datasets for future use, preferably in Parquet format. This will make your later experiments faster and more reproducible, here is why:
The raw interaction file is large. CSV is convenient for distribution, but inefficient for repeated Spark jobs. One of your first steps should be to convert the interaction file to Parquet and use the Parquet version for all downstream computation.
You can use code like this to do so:
interactions = spark.read.csv(
    "hdfs:///user/pw44_nyu_edu/goodreads_interactions.csv",
    header=True,
    inferSchema=True
)

interactions.write.mode("overwrite").parquet(
    "hdfs:///user/[YOUR_NETID]_nyu_edu/goodreads/interactions.parquet"
)

Data splitting for recommender systems is more delicate than ordinary random train/test splitting. You should split the data in a way that preserves user histories. For validation and test users, make sure each user has some observed interactions available for generating recommendations and some held-out interactions available for evaluation. Remember: you cannot evaluate recommendations for a user with no observed history.

You may discard users with too few interactions to support meaningful evaluation, but you must document exactly what you did.

When prototyping your implementation, you should work on smaller user-based samples of the data before scaling to the full dataset. Do not downsample individual interactions independently. Instead, sample users and keep all relevant interactions for those users.

## What we would like you to build / do (all 5 deliverables are equally weighted)

## 1) Market segmentation

Market segmentation relies on similarity. We first want you to identify books that occupy similar positions in the reading market.

Find the top 100 pairs of books that are read by the most similar groups of users.

For the sake of simplicity, you can operationalize a book’s “market position” by the set of users who have read it, regardless of the numerical rating. Two books are similar if many of the same users have read both books.

We strongly recommend doing this with a MinHash-based algorithm.

Your final report should include:

the top 100 most similar book pairs;
a suitable estimate of similarity for each pair;
a brief explanation of your similarity measure;
and a description of how you made the computation scalable.

## 2) Recommendations with the Popularity baseline 
Before implementing a sophisticated recommender system, begin with making recommendations with a popularity baseline. This should be simple enough to implement with basic dataframe computations.

For example, you might recommend:

the most frequently read books;
the most frequently rated books;
the most frequently highly rated books;
or books ranked by a rating-weighted popularity score.

Your popularity baseline should be evaluated on the validation and test data using the same general evaluation framework you will use for later models.

Your report should clearly document:

how your popularity score was defined;
how recommendations were generated;
and how the popularity baseline performed.

## 3) Explicit-feedback ALS recommender

Your first latent-factor recommendation model should use Spark’s alternating least squares method to learn latent factor representations for users and books.

This model should be based strictly on explicit feedback: the numerical Goodreads ratings.

That means the model should use rating as the preference signal. For this deliverable, do not use is_read or is_reviewed as model inputs.

The ALS model has several hyperparameters that you should tune on the validation set, notably:

the rank, meaning the dimension of the latent factors;
and the regularization parameter.

Once you are able to make predictions, evaluate the model on the validation and test data. Scores for validation and test should both be reported in your final write-up.

Evaluations should be based on predictions of the top 100 books for each user, and should report appropriate ranking metrics. You may also additionally report RMSE, but ranking metrics should be central to your evaluation.

Your report should document:

your train/validation/test split;
your choice of hyperparameters;
your chosen evaluation metric;
validation performance of the recommender system;
test performance of the recommender system;
and how the explicit-feedback model compares to the popularity baseline.

As we discussed above, evaluations should be based on predictions of the top 100 items for each user, and report the ranking metrics provided by spark.
Refer to the [ranking metrics](https://spark.apache.org/docs/3.0.1/mllib-evaluation-metrics.html#ranking-systems) section of the Spark documentation for more details.

The choice of evaluation criteria for hyper-parameter tuning is up to you, as is the range of hyper-parameters you consider, but be sure to document your choices in the final report.
As a general rule, you should explore ranges of each hyper-parameter that are sufficiently large to produce observable differences in your evaluation score.
If you like, you may also use additional software implementations of recommendation or ranking metric evaluations, but be sure to cite any additional software you use in the project.


## 4) Implicit-feedback ALS recommender

Now, build a recommender system based only on implicit feedback.

For this deliverable, do not use the numerical ratings as the preference signal. Instead, construct a model from behavioral information such as whether the user read or reviewed the book.

For example, you might use:

is_read

as a binary implicit preference signal, and perhaps use:

is_reviewed

as an additional engagement or confidence signal.

The key conceptual distinction is:

is_read      = evidence of consumption
is_reviewed  = evidence of engagement

Neither field necessarily means that the user liked the book. A book can be read without necessarily having a useful numerical rating, and a review does not imply a positive rating.

Evaluate this implicit-feedback model using the same validation and test framework as the explicit-feedback model.

Your report should explain:

how you used implicit feedback as model inputs;
whether you used Spark ALS in implicit-feedback mode;
how you interpreted is_read and is_reviewed;
and how the implicit-feedback model compares to the explicit-feedback model from deliverable 3.


## 5) Combined explicit + implicit feedback model

Finally, build a model that combines explicit and implicit feedback into a single recommendation system.

For this deliverable, you should use the implicit-feedback fields to transform, weight, or synthesize a new rating-like or confidence-like signal.

For example, you might combine:

the numerical rating;
whether the user marked the book as read;
whether the user reviewed the book;
or other transformations of these fields that you can justify.

One simple approach would be to construct a combined score such as:

combined_score = f(rating, is_read, is_reviewed)

Another approach would be to treat the rating as the explicit preference signal, while using is_read and is_reviewed to modify the confidence assigned to the interaction.

You are not expected to invent a new recommender algorithm from scratch. You may reuse Spark ALS. The main requirement is that your model uses both explicit and implicit information in a principled way.

Your report should clearly explain:

how explicit and implicit feedback were combined;
why your transformation is reasonable;
what hyperparameters or weighting choices you considered;
and whether the combined model improved performance relative to:
the popularity baseline;
the explicit-feedback ALS model;
and the implicit-feedback ALS model.

### Using the cluster

Please be considerate of your fellow classmates!
The Dataproc cluster is a limited, shared resource. 
Make sure that your code is properly implemented and works efficiently. 
If too many people run inefficient code simultaneously, it can slow down the entire cluster for everyone.


## What to turn in

In addition to all of your code, produce a final report (no more than 5 pages), describing your implementation, answer to questions and evaluation results.
Your report should clearly identify the contributions of each member of your group, as well as AI contributions.
If any additional software components were required in your project, your choices should be described and well motivated here.  

Include a PDF of your final report through Brightspace.  Specifically, your final report should include the following details:

- Link to your group's GitHub repository
- List of top 100 most similar pairs (include a suitable estimate of their similarity for each pair), sorted by similarity
- Documentation of how your train/validation splits were generated
- Any additional pre-processing of the data that you decide to implement
- Evaluation of popularity baseline
- Documentation of latent factor models hyper-parameters and validation
- Evaluation of latent factor models

Any additional software components that you use should be cited and documented with installation instructions.


