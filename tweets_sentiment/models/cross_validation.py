import json
import train_and_eval as te
from tweets_sentiment.preprocessing.constants import FULL_PATH

from os import path
from sklearn.model_selection import GridSearchCV

def save_to_file(classifier_name, clf):
    filepath = path.join(FULL_PATH, 'data/%s_parameters.json'%classifier_name)
    with open(filepath, 'w') as f:
        json.dump(clf.best_params_, f)


def perform_cross_validation(classifier, parameters, classifier_name, feature_vector, y_train):
    print('===> Estimating parameters')
    clf = GridSearchCV(classifier, parameters, verbose=2)
    te.train(clf, feature_vector, y_train)
    save_to_file(classifier_name, clf)
    print('===> Best params for: %s:'%classifier_name)
    print(clf.best_params_)
    return clf.best_params_
