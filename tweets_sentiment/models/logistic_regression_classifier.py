import embedding as we
import train_and_eval as te
import pandas as pd
import numpy as np
import cross_validation as cv
import json

from os import path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET
from tweets_sentiment.preprocessing.constants import FULL_PATH
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def init_logistic_regression(params=None):
    if params:
        return LogisticRegression(params)
    else:
        return LogisticRegression()


def read_data():
    data = pd.read_csv(PREPROCESSED_DATASET)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return labels, tweets


if __name__ == '__main__':
    logistic_regression = init_logistic_regression()

    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.33)
    feature_vector, vectorizer = we.make_bag_of_words(X_train)

    # params = estimate_parameters(svm, feature_vector.toarray(), y_train)
    # params = read_params()
    # logistic_regression.set_params(**params)

    classifier = te.train(logistic_regression, feature_vector.toarray(), y_train)

    transformed_tweets = vectorizer.transform(X_test).toarray()
    y_true, y_pred = y_test, classifier.predict(transformed_tweets)
    te.evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test)
