from os import path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

import pandas as pd
import words_embedding as we
import train_and_eval as te

full_path = path.dirname(path.abspath(__file__ + "/../"))
dataset = path.join(full_path, 'data/preprocessed_dataset.csv')


def init_naive_bayes():
    return MultinomialNB()


def get_sentiment(classifier, vectorizer, tweet):
    transformed_tweet = vectorizer.transform([tweet]).toarray()
    return classifier.predict(transformed_tweet)


def read_data():
    data = pd.read_csv(dataset)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return labels, tweets


if __name__ == '__main__':
    nb = init_naive_bayes()
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.33)
    feature_vector, vectorizer = we.make_bag_of_words(tweets)
    classifier = te.train(nb, X_train, y_train)
    te.evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test)
