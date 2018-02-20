from sklearn.metrics import classification_report


def train(classifier, feature_vector, labels):
    classifier.fit(feature_vector, labels)
    return classifier


def compute_accuracy(classifier, vectorizer, tweets, labels):
    transformed_tweets = vectorizer.transform(tweets).toarray()
    prediction = classifier.predict(transformed_tweets)

    target_names = ['negative', 'positive']
    report = classification_report(labels, prediction, target_names=target_names)
    print(report)

    score = classifier.score(transformed_tweets, labels)
    print('Score: ' + str(score))


def evaluate(classifier, vectorizer, X_train, X_test, y_train, y_test):
    print('Training set accuracy: ')
    compute_accuracy(classifier, vectorizer, X_train, y_train)
    print('Test set accuracy: ')
    compute_accuracy(classifier, vectorizer, X_test, y_test)
