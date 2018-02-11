from os import path
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
import nltk.tokenize
from constants import POSITIVE_EMOTICONS, NEGATIVE_EMOTICONS, POSITIVE_WORD, NEGATIVE_WORD, SHORT_WORDS


SLANG_FILE_PATH = 'data/slang.txt'


def init_tokenizer(preserve_case=False, strip_handles=False, reduce_len=True):
    return nltk.tokenize.TweetTokenizer(preserve_case,
                                        strip_handles,
                                        reduce_len)


def transform_post(twitter_post):
    tokens = []
    tokenizer = init_tokenizer()
    slang_dictionary = load_sleng_dict()
    for token in tokenizer.tokenize(twitter_post):
        process_token = transform_slang_words(slang_dictionary,
                                              emoticon_transformation(token))
        tokens.append(process_token)

    pos_tag_tokens = pos_tagging(tokenizer.tokenize(' '.join(tokens)))
    return lemmatization(pos_tag_tokens)


def lemmatization(pos_tag_sentence):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []
    for pos_tuple in pos_tag_sentence:
        lemmatized_words.append(lemmatizer.lemmatize(pos_tuple[0], transform_tag(pos_tuple[1])))

    return ' '.join(lemmatized_words)


def pos_tagging(tokenized_post):
    return nltk.pos_tag(tokenized_post)


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


def emoticon_transformation(token):
    if token in POSITIVE_EMOTICONS:
        return POSITIVE_WORD
    elif token in NEGATIVE_EMOTICONS:
        return NEGATIVE_WORD

    return token


def transform_slang_words(slang_dictionary, token):
    if token in slang_dictionary:
        return slang_dictionary[token]
    else:
        return token


def load_sleng_dict():
    basepath = path.dirname(path.abspath(__file__ + "/../"))
    full_path = path.join(basepath, SLANG_FILE_PATH)
    slang_dictionary = {}
    with open(full_path, 'r') as slang_file:
        for line in slang_file:
            splits = line.replace('\t', ' ').split(' ', 1)
            slang_dictionary.update({splits[0]: splits[1].strip()})

    return slang_dictionary
