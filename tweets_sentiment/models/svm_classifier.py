from sklearn.linear_model import SGDClassifier

import words_embedding as we
import train_and_eval as te


def init_svm():
    return SGDClassifier()


if __name__ == '__main__':
    svm = init_svm()
    labels, tweets = te.load_training_data()
    feature_vector, vectorizer = we.make_bag_of_words(tweets)
    classifier = te.train(svm, feature_vector.toarray(), labels)
    te.evaluate(classifier, vectorizer)
