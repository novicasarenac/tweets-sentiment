import embedding as we
import train_and_eval as te
import pandas as pd
import numpy as np
import cross_validation as cv
import json

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


def init_svm(params=None):
    if params:
        return LinearSVC(params)
    else:
        return LinearSVC()

    # return LinearSVC(max_iter=1500, verbose=15, C=0.01)


def read_data():
    data = pd.read_csv(PREPROCESSED_DATASET)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return labels, tweets


def estimate_parameters(svm, feature_vector, y_train):
    tuned_parameters = [
        {'C': np.linspace(0.001, 1.0, 50),
        'max_iter': [1000,1500,2000]},
    ]
    return cv.perform_cross_validation(svm, tuned_parameters, 'svm', feature_vector, y_train)


def read_params():
    with open('svm_parameters.json', 'r') as f:
        parameters = json.load(f)
    return parameters


if __name__ == '__main__':
    svm = init_svm()

    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.33)
    feature_vector, vectorizer = we.make_bag_of_words(X_train)

    # params = estimate_parameters(svm, feature_vector.toarray(), y_train)
    params = read_params()
    svm.set_params(**params)

    classifier = te.train(svm, feature_vector.toarray(), y_train)

    transformed_tweets = vectorizer.transform(X_test).toarray()
    y_true, y_pred = y_test, classifier.predict(transformed_tweets)
    te.evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test)
