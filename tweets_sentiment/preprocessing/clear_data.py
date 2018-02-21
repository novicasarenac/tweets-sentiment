import re

from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import RegexpTokenizer
from pipe import Pipe


@Pipe
def remove_stopwords(tokenized_text):
    return [word for word in tokenized_text if word not in stopwords.words('english')]


@Pipe
def remove_urls(tokenized_text):
    reg = re.compile('http[s]?://|www.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return [word for word in tokenized_text if not reg.match(word)]


@Pipe
def remove_numbers(tokenized_text):
    return [word for word in tokenized_text if not word.isdigit()]


@Pipe
def remove_multiple_occurrence(tokenized_text):
    return [re.sub(r'(.)\1+', r'\1\1', word) for word in tokenized_text]


@Pipe
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
    tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    clean_tweet = tokens \
            | remove_urls \
            | process_hashtags \
            | remove_stopwords \
            | remove_numbers \
            | remove_multiple_occurrence
    return ' '.join(clean_tweet)


if __name__ == "__main__":
    example = "she has the same 10 :() car Volvo : as @Marco ooookkkkk #ferenc #car http://www.google.com"
    processed_ex = clear_data(example)
    print(remove_special_characters(processed_ex))
