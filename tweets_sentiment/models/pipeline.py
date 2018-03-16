from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


def bag_of_words(classifier, tf_idf=False):
    if tf_idf:
        steps = [('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('clf', classifier),
                 ]
    else:
        steps = [('vect', CountVectorizer()),
                 ('clf', classifier),
                 ]
    return Pipeline(steps)
