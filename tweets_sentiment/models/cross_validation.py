import json
from tweets_sentiment.preprocessing.constants import FULL_PATH

from os import path
from sklearn.model_selection import GridSearchCV


def search_params(pipeline, parameters, classifier_name, feature_vector, y_train):
    print('===> Estimating parameters')
    clf = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=3)
    clf.fit(feature_vector, y_train)
    save_to_file(classifier_name, clf)
    print('===> Best params for: {}'.format(classifier_name))
    print(clf.best_params_)
    print('===> Best score: {:.2f}'.format(clf.best_score_))
    return clf.best_params_


def save_to_file(classifier_name, clf):
    filepath = path.join(FULL_PATH, 'data/{}_parameters.json'.format(classifier_name))
    with open(filepath, 'w') as f:
        json.dump(clf.best_params_, f)
