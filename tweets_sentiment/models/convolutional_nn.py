import numpy as np
import pandas as pd
import h5py

from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.constants import GLOVE_PATH
from tweets_sentiment.preprocessing.constants import LARGE_DATASET_RAW
from tweets_sentiment.preprocessing.constants import CNN_MODEL
from tweets_sentiment.preprocessing.constants import CNN_WEIGHTS


EMBEDDING_DIMENSION = 100


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


def read_dataset():
    data = pd.read_csv(LARGE_DATASET_RAW, error_bad_lines=False)

    labels = data['Sentiment']
    tweets = data['SentimentText']

    return tweets, labels


def tokenize_dataset(tweets):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    # dictionary word:index
    words_indices = tokenizer.word_index
    print('===> Number of words in dataset: {}'.format(len(words_indices)))

    return sequences, words_indices


# mapping pretrained coefficients to dataset
def create_embedding_matrix(word_indices, embeddings):
    WORDS_NUM = len(word_indices) + 1
    embedding_matrix = np.zeros((WORDS_NUM, EMBEDDING_DIMENSION))
    for word, i in word_indices.items():
        word_vector = embeddings.get(word)
        if word_vector is not None:
            embedding_matrix[i] = word_vector

    return WORDS_NUM, embedding_matrix


def get_model(WORDS_NUM, embedding_matrix, MAX_SEQUENCE_LENGTH):
    model = Sequential()
    model.add(Embedding(WORDS_NUM, EMBEDDING_DIMENSION, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
    model.add(Conv1D(256, 3, padding='same', activation='relu'))
    model.add(Conv1D(128, 3, padding='same', activation='relu'))
    model.add(Conv1D(64, 3, padding='same'))
    model.add(GlobalMaxPooling1D())
    # model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(180, activation='sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    return model


def save_model_and_weights(model):
    print('===> Saving model and weights\n')
    model_json = model.to_json()
    with open(CNN_MODEL, 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(CNN_WEIGHTS)


def train_cnn():
    print('===> Reading GloVe words embeddings\n')
    embeddings = read_embeddings()
    # 1.6M tweets
    tweets, labels = read_dataset()
    sequences, word_indices = tokenize_dataset(tweets)

    MAX_SEQUENCE_LENGTH = len(max(sequences, key=lambda x: len(x)))
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    # labels = to_categorical(labels)
    print('===> Data shape: {}\n'.format(padded_sequences.shape))
    print('===> Labels shape: {}\n'.format(labels.shape))

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences,
                                                        labels,
                                                        test_size=0.2)

    WORDS_NUM, embedding_matrix = create_embedding_matrix(word_indices,
                                                          embeddings)

    model = get_model(WORDS_NUM, embedding_matrix, MAX_SEQUENCE_LENGTH)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=5, validation_data=(X_test, y_test), verbose=1)

    # Evaluation on the test set
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores[1])

    save_model_and_weights(model)


if __name__ == '__main__':
    train_cnn()
