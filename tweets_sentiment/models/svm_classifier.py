import embedding as we
import train_and_eval as te
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET


def init_svm():
    return LinearSVC(max_iter=1500, verbose=15, C=0.01)


def read_data():
    data = pd.read_csv(PREPROCESSED_DATASET)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return labels, tweets


if __name__ == '__main__':
    svm = init_svm()
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.33)
    feature_vector, vectorizer = we.make_bag_of_words(X_train)
    classifier = te.train(svm, feature_vector.toarray(), y_train)
    te.evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test)
