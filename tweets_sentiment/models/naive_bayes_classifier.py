from sklearn.naive_bayes import MultinomialNB
from tweets_sentiment.data_preprocessing import preprocess_data as pd
import words_embedding as we

data = '../../data/testData.csv'


def train():
    labels, tweets = pd.vector_representation(data)
    bag_of_words, vectorizer = we.make_bag_of_words(tweets)
    classifier = MultinomialNB()
    classifier.fit(bag_of_words.toarray(), labels)
    return classifier, vectorizer


if __name__ == '__main__':
    clf, vectorizer = train()
    example = "It was good last night"
    transformed_example = vectorizer.transform([example]).toarray()
    a = clf.predict(transformed_example)
    print(a)
