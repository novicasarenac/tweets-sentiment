from sklearn.naive_bayes import MultinomialNB
import data_preprocessing as dp
import words_embedding as we

data = '../../data/testData.csv'

def train():
    labels, data = dp.vector_representation(data)
    bagOfWords = we.makeBagOfWords(data)
    classifier = MultinomialNB().fit(bagOfWords, labels)
    return classifier

if __name__ == '__main__':
    train()
    clf = train()
    example = "I am so happy :)"
    print(clf.predict(makeBagOfWords(example)))
