from sklearn.metrics import classification_report


def compute_accuracy(pipeline, tweets, labels):
    prediction = pipeline.predict(tweets)

    target_names = ['Negative', 'Positive']
    report = classification_report(labels, prediction, target_names=target_names)
    print(report)

    score = pipeline.score(tweets, labels)
    print('Score: ' + str(score))


def evaluate(pipeline, X_train, X_test, y_train, y_test):
    print('===> Training set accuracy: ')
    compute_accuracy(pipeline, X_train, y_train)
    print('===> Test set accuracy: ')
    compute_accuracy(pipeline, X_test, y_test)
