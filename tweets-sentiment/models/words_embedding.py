from sklearn.feature_extraction.text import CountVectorizer

def makeBagOfWords(tweets):
    vectorizer = CountVectorizer()
    bagOfWords = vectorizer.fit_transform(tweets)
    return bagOfWords

if __name__ == '__main__':
    example = ['this is the first', 'this is the second']
    print(makeBagOfWords(example).toarray())
