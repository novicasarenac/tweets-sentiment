import numpy as np

from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from embedding import read_embeddings
from embedding import tokenize_dataset
from sklearn.model_selection import train_test_split
from tweets_sentiment.preprocessing.preprocess import read_corpus_dataset
from tweets_sentiment.preprocessing.constants import LARGE_DATASET_RAW
from tweets_sentiment.preprocessing.constants import CNN_MODEL
from tweets_sentiment.preprocessing.constants import CNN_WEIGHTS


EMBEDDING_DIMENSION = 100
VOCABULARY_SIZE = 50000


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
    model.add(Embedding(WORDS_NUM,
                        EMBEDDING_DIMENSION,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))
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
    tweets, labels = read_corpus_dataset(LARGE_DATASET_RAW)
    sequences, word_indices = tokenize_dataset(tweets, VOCABULARY_SIZE)

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
    model.fit(X_train,
              y_train,
              batch_size=128,
              epochs=5,
              validation_data=(X_test, y_test),
              verbose=1)

    # Evaluation on the test set
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores[1])

    save_model_and_weights(model)


if __name__ == '__main__':
    train_cnn()
