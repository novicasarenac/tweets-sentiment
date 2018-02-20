from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from os import path
import words_embedding as we
import pandas as pd
from sklearn.model_selection import train_test_split

full_path = path.dirname(path.abspath(__file__ + "/../"))
new_data = path.join(full_path, 'data/preprocessed_dataset.csv')


def train(labels, tweets):
    bag_of_words, vectorizer = we.make_bag_of_words(tweets)
    classifier = MultinomialNB()
    classifier.fit(bag_of_words.toarray(), labels)
    return classifier, vectorizer


def compute_accuracy(classifier, vectorizer, tweets, labels):
    transformed_tweets = vectorizer.transform(tweets).toarray()
    prediction = classifier.predict(transformed_tweets)
    # precision, recall, f1 score
    target_names = ['negative', 'positive']
    report = classification_report(labels, prediction, target_names = target_names)
    print(report)
    # classifier score
    classifier_score = classifier.score(transformed_tweets, labels)
    print('Score: ' + str(classifier_score))


def evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test):
    print('Training set accuracy: ')
    compute_accuracy(classifier, vectorizer, X_train, y_train)
    print('Test set accuracy: ')
    compute_accuracy(classifier, vectorizer, X_test, y_test)


def get_sentiment(classifier, vectorizer, tweet):
    transformed_tweet = vectorizer.transform([tweet]).toarray()
    return classifier.predict(transformed_tweet)


def read_data():
    data = pd.read_csv(new_data)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return labels, tweets


if __name__ == '__main__':
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size = 0.33)

    classifier, vectorizer = train(y_train, X_train)
    evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test)
