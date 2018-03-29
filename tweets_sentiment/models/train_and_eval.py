import json
import pandas as pd

from os import path
from pipeline import bag_of_words
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.constants import FULL_PATH
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET


def read_data():
    data = pd.read_csv(PREPROCESSED_DATASET)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return labels, tweets


def read_params(clf_params_name):
    filepath = path.join(FULL_PATH, clf_params_name)
    with open(filepath, 'r') as f:
        parameters = json.load(f)
    return parameters


def compute_accuracy(pipeline, tweets, labels):
    prediction = pipeline.predict(tweets)

    target_names = ['Negative', 'Positive']
    report = classification_report(labels,
                                   prediction,
                                   target_names=target_names)
    print(report)

    score = pipeline.score(tweets, labels)
    print('Score: ' + str(score))


def evaluate(pipeline, X_train, X_test, y_train, y_test):
    print('===> Training set accuracy: ')
    compute_accuracy(pipeline, X_train, y_train)
    print('===> Test set accuracy: ')
    compute_accuracy(pipeline, X_test, y_test)


def load_pipelines(tf_idf):
    print("===> Loading all pipelines...")

    nb_pipeline = bag_of_words(MultinomialNB(), tf_idf)
    lr_pipeline = bag_of_words(LogisticRegression(), tf_idf)
    svm_pipeline = bag_of_words(LinearSVC(), tf_idf)
    xgb_pipeline = bag_of_words(XGBClassifier(), tf_idf)

    print("===> Setting parameters...")
    nb_pipeline.set_params(**read_params("data/nb_parameters.json"))
    lr_pipeline.set_params(**read_params("data/lr_parameters.json"))
    svm_pipeline.set_params(**read_params("data/svm_parameters.json"))
    xgb_pipeline.set_params(**read_params("data/xgb_parameters.json"))

    pipelines = {'Naive Bayes': nb_pipeline,
                 'Logistic Regression': lr_pipeline,
                 'Linear SVM': svm_pipeline,
                 'XGBooster': xgb_pipeline,
                 }

    return pipelines


if __name__ == "__main__":
    # load classifier pipelines with or without Tfidf transformer
    pipelines = load_pipelines(False)
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets,
                                                        labels,
                                                        test_size=0.2)
    print("===> Fitting pipelines:")
    for key, value in pipelines.items():
        value.fit(X_train, y_train)

    print("===> Evaluating pipelines:")
    for key, value in pipelines.items():
        print("\n")
        print("{:>15}:".format(key))
        evaluate(value, X_train, X_test, y_train, y_test)
