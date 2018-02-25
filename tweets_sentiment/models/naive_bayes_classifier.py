import pandas as pd
import embedding as we
import train_and_eval as te
import numpy as np
import cross_validation as cv
import json

from os import path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET
from tweets_sentiment.preprocessing.constants import FULL_PATH


def init_naive_bayes():
    return MultinomialNB()


def get_sentiment(classifier, vectorizer, tweet):
    transformed_tweet = vectorizer.transform([tweet]).toarray()
    return classifier.predict(transformed_tweet)


def read_data():
    data = pd.read_csv(PREPROCESSED_DATASET)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return labels, tweets


def estimate_parameters(nb, feature_vector, y_train):
    tuned_parameters = [
        {'alpha': np.linspace(0.01, 5.0, 180)},
    ]
    return cv.perform_cross_validation(nb, tuned_parameters, 'nb', feature_vector, y_train)


def read_params():
    filepath = path.join(FULL_PATH, 'data/nb_parameters.json')
    with open(filepath, 'r') as f:
        parameters = json.load(f)
    return parameters


if __name__ == '__main__':
    nb = init_naive_bayes()
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.33)
    feature_vector, vectorizer = we.make_bag_of_words(X_train)

    # params = estimate_parameters(nb, feature_vector.toarray(), y_train)
    params = read_params()
    nb.set_params(**params)

    classifier = te.train(nb, feature_vector.toarray(), y_train)
    te.evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test)
