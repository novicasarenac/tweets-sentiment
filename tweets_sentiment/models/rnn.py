import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.constants import WORD2VEC_MODEL
from tweets_sentiment.preprocessing.constants import LARGE_DATASET_DESTINATION

EMBEDDING_DIM = 300
CELL_UNITS = 128
BATCH_SIZE = 32
VOCABULARY_SIZE = 40000


def read_dataset():
    data = pd.read_csv(LARGE_DATASET_DESTINATION)
    labels = data['sentiment']
    tweets = data['tweet'].values.astype('U')

    return tweets, labels


def tokenize_dataset(tweets):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    # dictionary word:index
    word_indices = {}
    for key, value in tokenizer.word_index.items():
        word_indices[key] = value
        if value == VOCABULARY_SIZE:
            break
    print('===> Number of words in dataset: {}'.format(len(word_indices)))

    return sequences, word_indices


def create_embedding_matrix(word_indices, w2v_model):
    words_num = len(word_indices) + 1
    embedding_matrix = np.zeros((words_num, EMBEDDING_DIM))
    for word, i in word_indices.items():
        if word in w2v_model.vocab:
            embedding_matrix[i] = w2v_model.word_vec(word)

    return words_num, embedding_matrix


def create_model(vocab_size, embedding_matrix, MAX_SEQUENCE_LENGTH):
    model = Sequential()
    model.add(Embedding(vocab_size,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))
    model.add(LSTM(CELL_UNITS,
                   dropout=0.4,
                   recurrent_dropout=0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model


def train_model():
    print('===> Reading word2vec model...')
    word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL,
                                                       binary=True)

    tweets, labels = read_dataset()
    word_sequences, word_indices = tokenize_dataset(tweets)
    vocab_size, embedding_matrix = create_embedding_matrix(word_indices,
                                                           word2vec_model)

    MAX_SEQUENCE_LENGTH = len(max(word_sequences, key=lambda x: len(x)))
    padded_words = pad_sequences(word_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    X_train, X_test, y_train, y_test = train_test_split(padded_words,
                                                        labels,
                                                        test_size=0.2)

    model = create_model(vocab_size,
                         embedding_matrix,
                         MAX_SEQUENCE_LENGTH)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    print('===> Start training neural network...')
    model.fit(X_train,
              y_train,
              batch_size=BATCH_SIZE,
              epochs=3,
              validation_split=0.2, verbose=1)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores[1])


if __name__ == "__main__":
    train_model()
