from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
import re

def remove_stopwords(tokenizedText):
    return [word for word in tokenizedText if word not in stopwords.words('english')]

def remove_urls(tokenizedText):
    reg = re.compile('http[s]?://|www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return [word for word in tokenizedText if not reg.match(word)]

def process_hashtags(tokenizedText):
    reg = re.compile('(?:^|\s)[＃#]{1}(\w+)')
    withoutHashtags = []
    for word in tokenizedText:
        if reg.match(word):
            withoutHashtags.append(word.replace('#', ''))
        else:
            withoutHashtags.append(word)
    return withoutHashtags

def clear_data(tweet):
    #remove tags and reduce length
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    withoutUrls = remove_urls(tokens)
    withProcessedHashtags = process_hashtags(withoutUrls)
    cleanTweet = remove_stopwords(withProcessedHashtags)
    print(cleanTweet)

if __name__ == "__main__":
    clear_data("she has the same car Volvo as @Marco ooookkkkk #ferenc #car http://www.google.com");
