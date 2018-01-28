from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
import re


def remove_stopwords(tokenized_text):
    return [word for word in tokenized_text if word not in stopwords.words('english')]


def remove_urls(tokenized_text):
    reg = re.compile('http[s]?://|www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return [word for word in tokenized_text if not reg.match(word)]


def process_hashtags(tokenized_text):
    reg = re.compile('(?:^|\s)[＃#]{1}(\w+)')
    without_hashtags = []
    for word in tokenized_text:
        if reg.match(word):
            without_hashtags.append(word.replace('#', ''))
        else:
            without_hashtags.append(word)
    return without_hashtags


def remove_special_characters(tweet):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(tweet)
    return ' '.join(tokens)


def clear_data(tweet):
    #remove tags and reduce length
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    without_urls = remove_urls(tokens)
    with_processed_hashtags = process_hashtags(without_urls)
    clean_tweet = remove_stopwords(with_processed_hashtags)
    return ' '.join(clean_tweet)


if __name__ == "__main__":
    example = "she has the same :() car Volvo : as @Marco ooookkkkk #ferenc #car http://www.google.com"
    processed_ex = clear_data(example)
    print(remove_special_characters(processed_ex))
