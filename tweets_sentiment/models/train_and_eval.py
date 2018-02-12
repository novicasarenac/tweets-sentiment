from os import path
from sklearn.metrics import classification_report
from tweets_sentiment.data_preprocessing import preprocess_data as pd

import words_embedding as we

full_path = path.dirname(path.abspath(__file__ + "/../"))
training_set = path.join(full_path, 'data/preprocessedData.csv')
test_set = path.join(full_path, 'data/testData.csv')


def load_training_data():
    return pd.vector_representation(training_set)

def train(classifier, feature_vector, labels):
    classifier.fit(feature_vector, labels)
    return classifier

def compute_accuracy(classifier, vectorizer, data_set):
    labels, tweets = pd.vector_representation(data_set)
    transformed_tweets = vectorizer.transform(tweets).toarray()
    prediction = classifier.predict(transformed_tweets)

    target_names = ['negative', 'positive']
    report = classification_report(labels, prediction, target_names = target_names)
    print(report)

    score = classifier.score(transformed_tweets, labels)
    print('Score: ' + str(score))


def evaluate(classifier, vectorizer):
    print('Training set accuracy: ')
    compute_accuracy(classifier, vectorizer, training_set)
    print('Test set accuracy: ')
    compute_accuracy(classifier, vectorizer, test_set)
