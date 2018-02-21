from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET

import pandas as pd
import words_embedding as we
import train_and_eval as te


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


if __name__ == '__main__':
    nb = init_naive_bayes()
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.33)
    feature_vector, vectorizer = we.make_bag_of_words(X_train)
    classifier = te.train(nb, feature_vector.toarray(), y_train)
    te.evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test)
