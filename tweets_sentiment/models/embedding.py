from sklearn.feature_extraction.text import CountVectorizer


def make_bag_of_words(tweets):
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=30000)
    bag_of_words = vectorizer.fit_transform(tweets)
    print('===> Features number:%s'%len(vectorizer.vocabulary_))
    return bag_of_words, vectorizer


if __name__ == '__main__':
    example = ['this is the first first', 'this is the second']
    bag_of_words, vectorizer = make_bag_of_words(example)
    print(bag_of_words.toarray())
