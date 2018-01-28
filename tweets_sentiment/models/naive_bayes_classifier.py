from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from tweets_sentiment.data_preprocessing import preprocess_data as pd
from os import path
import words_embedding as we

full_path = path.dirname(path.abspath(__file__ + "/../"))

training_set = path.join(full_path, 'data/preprocessedData.csv')
test_set = path.join(full_path, 'data/testData.csv')


def train():
    labels, tweets = pd.vector_representation(training_set)
    bag_of_words, vectorizer = we.make_bag_of_words(tweets)
    classifier = MultinomialNB()
    classifier.fit(bag_of_words.toarray(), labels)
    return classifier, vectorizer


def compute_accuracy(classifier, vectorizer, data_set):
    labels, tweets = pd.vector_representation(data_set)
    transformed_tweets = vectorizer.transform(tweets).toarray()
    prediction = classifier.predict(transformed_tweets)
    # precision, recall, f1 score
    target_names = ['negative', 'positive']
    report = classification_report(labels, prediction, target_names = target_names)
    print(report)
    # classifier score
    classifier_score = classifier.score(transformed_tweets, labels)
    print('Score: ' + str(classifier_score))


def evaluate(classifier, vectorizer):
    print('Training set accuracy: ')
    compute_accuracy(classifier, vectorizer, training_set)
    print('Test set accuracy: ')
    compute_accuracy(classifier, vectorizer, test_set)


def get_sentiment(classifier, vectorizer, tweet):
    transformed_tweet = vectorizer.transform([tweet]).toarray()
    return classifier.predict(transformed_tweet)


if __name__ == '__main__':
    classifier, vectorizer = train()
    evaluate(classifier, vectorizer)
    example = "It was good last night"
    print(get_sentiment(classifier, vectorizer, example))
