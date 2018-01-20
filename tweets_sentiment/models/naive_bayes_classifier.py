from sklearn.naive_bayes import MultinomialNB
from tweets_sentiment.data_preprocessing import preprocess_data as pd
import words_embedding as we

data = '../../data/testData.csv'


def train():
    labels, tweets = pd.vector_representation(data)
    bagOfWords = we.makeBagOfWords(tweets)
    classifier = MultinomialNB().fit(bagOfWords, labels)
    return classifier


if __name__ == '__main__':
    train()
    clf = train()
    example = "I am so happy :)"
    print(clf.predict(we.makeBagOfWords(example)))
