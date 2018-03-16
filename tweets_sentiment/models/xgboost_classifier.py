from pipeline import bag_of_words
from cross_validation import search_params
from train_and_eval import read_data
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split


def estimate_parameters(pipeline, feature_vector, y_train):
    parameters = {
            'vect__max_df': (0.25, 0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'clf__learning_rate': (0.1, 0.5, 1.0),
            'clf__n_estimators': (150, 200),
            'clf__booster': ('gbtree', 'gblinear', 'dart'),
        }

    search_params(pipeline, parameters, 'xgb', feature_vector, y_train)


if __name__ == '__main__':
    xgb_pipeline = bag_of_words(XGBClassifier())
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets,
                                                        labels,
                                                        test_size=0.33)
    estimate_parameters(xgb_pipeline, X_train, y_train)
