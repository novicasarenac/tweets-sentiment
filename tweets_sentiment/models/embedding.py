import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from tweets_sentiment.preprocessing.constants import WORD2VEC_MODEL
from tweets_sentiment.preprocessing.constants import GLOVE_PATH

WORDS_NUM = 5000


def tokenize_dataset(tweets, vocabulary_size):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    # dictionary word:index
    word_indices = {}
    for key, value in tokenizer.word_index.items():
        word_indices[key] = value
        if value == vocabulary_size:
            break
    print('===> Number of words in dataset: {}'.format(len(word_indices)))

    return sequences, word_indices


def char_tokenize_dataset(tweets):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    # dictionary character:index
    char_indices = tokenizer.word_index
    print('===> Number of characters in dataset: {}\n'.format(len(char_indices)))

    return sequences, char_indices


# read gloVe embeddings with 100 features
def read_embeddings():
    embeddings = {}
    with open(GLOVE_PATH) as glove_file:
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
    print('===> Using {} embedding vectors\n'.format(len(embeddings)))
    return embeddings


# load word2vec model with 300-dim features
def load_word2vec_model(w2v_path):
    print('===> Loading Word2Vec model...')
    return KeyedVectors.load_word2vec_format(WORD2VEC_MODEL, binary=True)


def visualize_word2vec(w2v_model, vocab_size):
    word_labels = []
    word_values = []

    for index, word in enumerate(w2v_model.wv.vocab):
        word_values.append(w2v_model[word])
        word_labels.append(word)
        print(index)
        if index == vocab_size:
            break

    print('===> Initializing TSNE model')
    tsne_model = TSNE(n_components=2, init='pca', verbose=1)
    print('===> Reducing dense vector dimension')
    fitted_values = tsne_model.fit_transform(word_values)

    x_points = []
    y_points = []

    for points in fitted_values:
        x_points.append(points[0])
        y_points.append(points[1])

    plt.figure(figsize=(32, 32))
    for i in range(len(x_points)):
        plt.scatter(x_points[i], y_points[i])
        plt.annotate(word_labels[i],
                     xy=(x_points[i], y_points[i]),
                     xytext=(5, 2),
                     textcoords='offset points')
    plt.show()


if __name__ == '__main__':
    word2vec_model = load_word2vec_model(WORD2VEC_MODEL)
    visualize_word2vec(word2vec_model, WORDS_NUM)
