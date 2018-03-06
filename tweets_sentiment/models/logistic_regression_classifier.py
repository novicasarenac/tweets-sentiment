from pipeline import make_pipeline
from cross_validation import search_params
from train_and_eval import read_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def estimate_parameters(lr_pipeline, feature_vector, y_train):
    parameters = {
            'vect__max_df': (0.25, 0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'clf__solver': ('liblinear', 'saga'),
            'clf__C': (0.01, 0.05, 0.1, 0.5, 1.0),
            'clf__max_iter': (150, 200),
            'clf__penalty': ('l1', 'l2'),
        }
    search_params(lr_pipeline, parameters, 'lr', feature_vector, y_train)


if __name__ == '__main__':
    logistic_pipeline = make_pipeline(LogisticRegression())
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets,
                                                        labels,
                                                        test_size=0.33)
    estimate_parameters(logistic_pipeline, X_train, y_train)
