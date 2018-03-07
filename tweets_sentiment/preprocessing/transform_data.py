import nltk
import nltk.tokenize

from os import path
from pipe import Pipe
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from tweets_sentiment.preprocessing.constants import POSITIVE_EMOTICONS
from tweets_sentiment.preprocessing.constants import NEGATIVE_EMOTICONS
from tweets_sentiment.preprocessing.constants import POSITIVE_WORD
from tweets_sentiment.preprocessing.constants import NEGATIVE_WORD
from tweets_sentiment.preprocessing.constants import SHORT_WORDS
from tweets_sentiment.preprocessing.constants import SLANG_FILE_PATH


def init_tokenizer(preserve_case=False, strip_handles=False, reduce_len=True):
    return nltk.tokenize.TweetTokenizer(preserve_case,
                                        strip_handles,
                                        reduce_len)


def transform_post(twitter_post, tokenizer, slang_dict, checker):
    tokens = tokenizer.tokenize(twitter_post)
    transformed_tweet = tokens \
            | emoticon_transformation \
            | transform_slang_words(slang_dict) \
            | transform_shortwords \
            | remove_special_characters \
            | spell_checker(checker) \
            | remove_one_character_words
            # | lemmatization
    return ' '.join(transformed_tweet)


@Pipe
def lemmatization(tokenized_text):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    pos_tag_sentence = nltk.pos_tag(tokenized_text)
    for pos_tuple in pos_tag_sentence:
        lemmatized_words.append(lemmatizer.lemmatize(pos_tuple[0], transform_tag(pos_tuple[1])))

    return lemmatized_words


@Pipe
def emoticon_transformation(tokenized_text):
    return [emoticon_check(token) for token in tokenized_text]


@Pipe
def transform_slang_words(tokenized_text, slang_dictionary):
    return [slang_dictionary[token] if token in slang_dictionary else token for token in tokenized_text]


@Pipe
def transform_shortwords(tokenized_text):
    return [SHORT_WORDS[token.lower()] if token.lower() in SHORT_WORDS else token for token in tokenized_text]


@Pipe
def spell_checker(tokenized_text, dictionary):
    return [check_dictionary(token, dictionary) for token in tokenized_text]


@Pipe
def remove_one_character_words(tokenized_text):
    return list(filter(None, [check_one_char_words(token) for token in tokenized_text]))


@Pipe
def remove_special_characters(tokenized_text):
    return [token.lower() for token in tokenized_text if token.isalpha()]


def load_sleng_dict():
    basepath = path.dirname(path.abspath(__file__ + "/../"))
    full_path = path.join(basepath, SLANG_FILE_PATH)
    slang_dictionary = {}
    with open(full_path, 'r') as slang_file:
        for line in slang_file:
            splits = line.replace('\t', ' ').split(' ', 1)
            slang_dictionary.update({splits[0]: splits[1].strip()})

    return slang_dictionary


def emoticon_check(token):
    if token in POSITIVE_EMOTICONS:
        return POSITIVE_WORD
    elif token in NEGATIVE_EMOTICONS:
        return NEGATIVE_WORD

    return token


def transform_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def check_dictionary(token, dictionary):
    if(dictionary.check(token)):
        return token
    else:
        suggest_arr = dictionary.suggest(token)
        return token if len(suggest_arr) == 0 else suggest_arr[0]


def check_one_char_words(token):
    if(len(token) == 1):
        return '' if "i" not in token else token

    return token
