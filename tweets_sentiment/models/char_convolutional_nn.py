from keras.models import Sequential
from keras.layers import Conv1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from embedding import char_tokenize_dataset
from tweets_sentiment.preprocessing.preprocess import read_corpus_dataset
from tweets_sentiment.preprocessing.constants import LARGE_DATASET_RAW
from tweets_sentiment.preprocessing.constants import CHAR_CNN_MODEL
from tweets_sentiment.preprocessing.constants import CHAR_CNN_WEIGHTS


EMBEDDING_DIMENSION = 16


def get_model(CHARS_NUM, MAX_SEQUENCE_LENGTH):
    model = Sequential()
    model.add(Embedding(CHARS_NUM,
                        EMBEDDING_DIMENSION,
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=True))
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
    tweets, labels = read_corpus_dataset(LARGE_DATASET_RAW)
    sequences, char_indices = char_tokenize_dataset(tweets)
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
    model.fit(X_train,
              y_train,
              batch_size=128,
              epochs=2,
              validation_split=0.1,
              verbose=1)

    # Evaluation on the test set
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(scores[1])

    save_model_and_weights(model)


if __name__ == '__main__':
    train_cnn()
