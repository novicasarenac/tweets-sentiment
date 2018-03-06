from pipeline import make_pipeline
from cross_validation import search_params
from train_and_eval import read_data
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split


def estimate_parameters(svm_pipeline, feature_vector, y_train):
    parameters = {
            'vect__max_df': (0.25, 0.5, 0.75, 1.0),
            'vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
            'clf__C': (0.001, 0.01, 0.05, 1.0),
            'clf__max_iter': (100, 150, 200),
            'clf__loss': ('hinge', 'squared_hinge'),
        }

    search_params(svm_pipeline, parameters, 'svm', feature_vector, y_train)


if __name__ == '__main__':
    svm_pipeline = make_pipeline(LinearSVC(), True)

    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets,
                                                        labels,
                                                        test_size=0.33)
    params = estimate_parameters(svm_pipeline, X_train, y_train)
