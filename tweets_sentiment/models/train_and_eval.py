import json
import pandas as pd
import numpy as np

from os import path
from numpy.random import shuffle
from pipeline import bag_of_words
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.preprocess import read_corpus_dataset
from tweets_sentiment.preprocessing.constants import FULL_PATH
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET


MIN_PARTITION = 20000
MAX_PARTITION = 1500000

NB_PARAMETERS = "data/nb_parameters.json"
LR_PARAMETERS = "data/lr_parameters.json"
SVM_PARAMETERS = "data/svm_parameters.json"
XGB_PARAMETERS = "data/xgb_parameters.json"


def read_partition(dataset_path, size):
    data = pd.read_csv(dataset_path, error_bad_lines=False)
    data = shuffle(data)
    partition = data.iloc[:size]
    tweets = partition['SentimentText'].values.astype('U')
    labels = partition['Sentiment']
    return tweets, labels


def read_params(clf_params_name):
    filepath = path.join(FULL_PATH, clf_params_name)
    with open(filepath, 'r') as f:
        parameters = json.load(f)
    return parameters


def compute_accuracy(pipeline, tweets, labels):
    prediction = pipeline.predict(tweets)

    target_names = ['Negative', 'Positive']
    report = classification_report(labels,
                                   prediction,
                                   target_names=target_names)
    print(report)

    score = pipeline.score(tweets, labels)
    print('Score: ' + str(score))


def evaluate(pipeline, X_train, X_test, y_train, y_test):
    print('===> Training set accuracy: ')
    compute_accuracy(pipeline, X_train, y_train)
    print('===> Test set accuracy: ')
    compute_accuracy(pipeline, X_test, y_test)


def load_pipeline(model, params_path=None, tf_idf=False, params=None):
    pipeline = bag_of_words(model, tf_idf)
    if params_path:
        pipeline.set_params(**read_params(params_path))
    elif params:
        pipeline.set_params(params)

    return pipeline


def set_pipelines(pipelines, key, value):
    pipelines = {key: value}
    return pipelines


def partition_corpus(min_examples, max_examples, n_partition=5):
    return np.geomspace(min_examples, max_examples, num=n_partition)


if __name__ == "__main__":
    # load classifier pipelines with or without Tfidf transformer
    pipelines = {}
    print('===> Loading pipelines')
    pipeline = load_pipeline(LogisticRegression(n_jobs=-1), LR_PARAMETERS)
    pipes = set_pipelines(pipelines, 'Logistic Regression', pipeline)

    partition_sizes = partition_corpus(MIN_PARTITION, MAX_PARTITION)
    partition_sizes = [int(round(x)) for x in partition_sizes]

    for index, partition_size in enumerate(partition_sizes):
        print('===> Iteration {}. Partition size: {}'.format(index + 1,
                                                             partition_size))
        tweets, labels = read_partition(PREPROCESSED_DATASET, partition_size)
        X_train, X_test, y_train, y_test = train_test_split(tweets,
                                                            labels,
                                                            test_size=0.2)
        print("===> Fitting pipelines:")
        for key, value in pipes.items():
            value.fit(X_train, y_train)

        print("===> Evaluating pipelines:")
        for key, value in pipes.items():
            print("\n")
            print("{:>15}:".format(key))
            evaluate(value, X_train, X_test, y_train, y_test)
