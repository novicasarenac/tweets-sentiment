from sklearn.feature_extraction.text import CountVectorizer


def make_bag_of_words(tweets):
    vectorizer = CountVectorizer()
    bag_of_words = vectorizer.fit_transform(tweets)
    return bag_of_words, vectorizer


if __name__ == '__main__':
    example = ['this is the first first', 'this is the second']
    print(make_bag_of_words(example).toarray())
