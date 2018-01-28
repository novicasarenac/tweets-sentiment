import nltk
import nltk.tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

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
    tokens = []
    for token in tokenizer.tokenize(twitter_post):
        tokens.append(emoticon_transformation(token))

    return ' '.join(tokens)


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
    if(token in POSITIVE_EMOTICONS):
        return POSITIVE_WORD
    elif(token in NEGATIVE_EMOTICONS):
        return NEGATIVE_WORD

    return token


if __name__ == "__main__":
    sentence = "going"
    tokenized_sentence = init_tokenizer().tokenize(sentence)
    pos_tagged = pos_tagging(tokenized_sentence)
    print(pos_tagged)
    print(lemmatization(pos_tagged))
