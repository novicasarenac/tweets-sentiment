import numpy as np
import pandas as pd
import h5py

from tweets_sentiment.preprocessing.constants import LARGE_DATASET_RAW
from tweets_sentiment.preprocessing.constants import CHAR_CNN_MODEL
from tweets_sentiment.preprocessing.constants import CHAR_CNN_WEIGHTS
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Conv1D, Dense, Dropout, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


EMBEDDING_DIMENSION = 16


def read_dataset():
    data = pd.read_csv(LARGE_DATASET_RAW, error_bad_lines=False)

    labels = data['Sentiment']
    tweets = data['SentimentText']

    return tweets, labels


def tokenize_dataset(tweets):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)
    # dictionary character:index
    char_indices = tokenizer.word_index
    print('===> Number of characters in dataset: {}\n'.format(len(char_indices)))

    return sequences, char_indices


def get_model(CHARS_NUM, MAX_SEQUENCE_LENGTH):
    model = Sequential()
    model.add(Embedding(CHARS_NUM, EMBEDDING_DIMENSION, input_length=MAX_SEQUENCE_LENGTH, trainable=True))
    model.add(Conv1D(128, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(64, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 5, padding='same'))
    model.add(MaxPooling1D())

    model.add(Flatten())
    # model.add(Dropout(0.4))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0))
    model.add(Dense(2, activation='softmax'))

    return model


def save_model_and_weights(model):
    print('===> Saving model and weights\n')
    model_json = model.to_json()
    with open(CHAR_CNN_MODEL, 'w') as json_file:
        json_file.write(model_json)

    model.save_weights(CHAR_CNN_WEIGHTS)


def train_cnn():
    tweets, labels = read_dataset()
    sequences, char_indices = tokenize_dataset(tweets)
    MAX_SEQUENCE_LENGTH = len(max(sequences, key=lambda x: len(x)))
    padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(labels)

    print('===> Data shape: {}\n'.format(padded_sequences.shape))
    print('===> Labels shape: {}\n'.format(labels.shape))

    X_train, X_test, y_train, y_test = train_test_split(padded_sequences,
                                                        labels,
                                                        test_size=0.2)

    CHARS_NUM = len(char_indices) + 1

    model = get_model(CHARS_NUM, MAX_SEQUENCE_LENGTH)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=128, epochs=2, validation_split=0.1, verbose=1)

    # Evaluation on the test set
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores[1])

    save_model_and_weights(model)


if __name__ == '__main__':
    train_cnn()
