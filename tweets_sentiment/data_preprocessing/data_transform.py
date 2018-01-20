import nltk.tokenize
from nltk.stem import PorterStemmer

POSITIVE_EMOTICONS = set([':)', '(:', ':]', '=]', ':D', ';)', ':-)', ':-]',
                          ':=]', ':-D', ':-))', '=)', ':-}', ':}', '=D'])

NEGATIVE_EMOTICONS = set([':(', '):', '=(', ':O', '=O', ':\\', ':-X', ':-|',
                          ':\'(', ':-\'(', ':[', ':-[', ':-(', ':@', ':P', ':-P'])

POSITIVE_WORD = 'happy'
NEGATIVE_WORD = 'sad'


def init_tokenizer(preserve_case=False, strip_handles=False, reduce_len=True):
    return nltk.tokenize.TweetTokenizer(preserve_case, strip_handles, reduce_len)


def transform_post(twitter_post):
    tokenizer = init_tokenizer()
    stemmer = PorterStemmer()
    tokens = []
    for token in tokenizer.tokenize(twitter_post):
        tokens.append(stemming(stemmer, emoticon_transformation(token)))

    return ' '.join(tokens)


def stemming(stemmer, token):
    return stemmer.stem(token)


def emoticon_transformation(token):
    if(token in POSITIVE_EMOTICONS):
        return POSITIVE_WORD
    elif(token in NEGATIVE_EMOTICONS):
        return NEGATIVE_WORD

    return token
