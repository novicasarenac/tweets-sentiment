import json
import pandas as pd

from os import path
from sklearn.metrics import classification_report
from tweets_sentiment.preprocessing.constants import FULL_PATH
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET


def read_data():
    data = pd.read_csv(PREPROCESSED_DATASET)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return labels, tweets


def read_params(clf_params_name):
    filepath = path.join(FULL_PATH, clf_params_name)
    with open(filepath, 'r') as f:
        parameters = json.load(f)
    return parameters


def compute_accuracy(pipeline, tweets, labels):
    prediction = pipeline.predict(tweets)

    target_names = ['Negative', 'Positive']
    report = classification_report(labels, prediction, target_names=target_names)
    print(report)

    score = pipeline.score(tweets, labels)
    print('Score: ' + str(score))


def evaluate(pipeline, X_train, X_test, y_train, y_test):
    print('===> Training set accuracy: ')
    compute_accuracy(pipeline, X_train, y_train)
    print('===> Test set accuracy: ')
    compute_accuracy(pipeline, X_test, y_test)
