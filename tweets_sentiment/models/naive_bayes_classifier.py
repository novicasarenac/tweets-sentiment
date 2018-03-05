from pipeline import make_pipeline
from cross_validation import search_params
from train_test_split import read_data
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def get_sentiment(classifier, vectorizer, tweet):
    transformed_tweet = vectorizer.transform([tweet]).toarray()
    return classifier.predict(transformed_tweet)


def estimate_parameters(nb_pipeline, feature_vector, y_train):
    parameters = {
        'vect__max_df': (0.25, 0.5, 0.75, 1.0),
        'vect__ngram_range': ((1, 1), (1, 2), (2, 2)),
        'clf__alpha': (0.01, 3.0, 10.0, 15.0),
        'clf__fit_prior': (True, False),
        }
    search_params(nb_pipeline, parameters, 'nb', feature_vector, y_train)


if __name__ == '__main__':
    nb_pipeline = make_pipeline(MultinomialNB())
    labels, tweets = read_data()
    X_train, X_test, y_train, y_test = train_test_split(tweets,
                                                        labels,
                                                        test_size=0.33)
    estimate_parameters(nb_pipeline, X_train, y_train)
