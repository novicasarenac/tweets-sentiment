import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json
from keras.layers.embeddings import Embedding
from keras.preprocessing.sequence import pad_sequences
from embedding import tokenize_dataset
from embedding import load_word2vec_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tweets_sentiment.preprocessing.preprocess import read_corpus_dataset
from tweets_sentiment.preprocessing.constants import WORD2VEC_MODEL
from tweets_sentiment.preprocessing.constants import LARGE_DATASET_RAW
from tweets_sentiment.preprocessing.constants import RNN_MODEL
from tweets_sentiment.preprocessing.constants import RNN_WEIGHTS

EMBEDDING_DIM = 300
CELL_UNITS = 128
BATCH_SIZE = 32
VOCABULARY_SIZE = 50000


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
    model.add(LSTM(CELL_UNITS))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def prepare_data(word2vec_model):
    print('===> Preparing data')
    tweets, labels = read_corpus_dataset(LARGE_DATASET_RAW)
    word_sequences, word_indices = tokenize_dataset(tweets, VOCABULARY_SIZE)
    vocab_size, embedding_matrix = create_embedding_matrix(word_indices,
                                                           word2vec_model)

    return word_sequences, vocab_size, embedding_matrix, labels


def train_model():
    word2vec_model = load_word2vec_model(WORD2VEC_MODEL)
    word_sequences, vocab_size, embedding_matrix, labels = prepare_data(word2vec_model)

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
    save_model(model)
    print('===> Training finished...')


def save_model(model):
    print('===> Saving model\n')
    model_json = model.to_json()
    with open(RNN_MODEL, 'w') as model_file:
        model_file.write(model_json)

    print('===> Saving model weights\n')
    model.save_weights(RNN_WEIGHTS)


def load_model(model_path, weights_path):
    print('===> Loading model from json')
    json_file = open(model_path, 'r')
    json_model_loaded = json_file.read()
    rnn_model = model_from_json(json_model_loaded)
    rnn_model.load_weights(weights_path)
    json_file.close()
    print('===> Model loaded')
    return rnn_model


def calculate_f1_score(model, test_data, train_labels):
    print('===> Calculating predictions')
    predictions = model.predict(test_data)
    rounded_predictions = [round(x[0]) for x in predictions]
    report = classification_report(train_labels,
                                   rounded_predictions,
                                   target_names=['Negative', 'Positive'])
    print(report)


if __name__ == "__main__":
    w2v_model = load_word2vec_model(WORD2VEC_MODEL)
    word_sequences, _, _, labels = prepare_data(w2v_model)

    max_length_seq = len(max(word_sequences, key=lambda x: len(x)))
    padded_sequence = pad_sequences(word_sequences, maxlen=max_length_seq)
    _, X_test, _, y_test = train_test_split(padded_sequence,
                                            labels,
                                            test_size=0.2)

    try:
        model = load_model(RNN_MODEL, RNN_WEIGHTS)
        print('Writing model summary:\n')
        print(model.summary())
        calculate_f1_score(model, X_test, y_test)
    except Exception:
        print('===> Model not saved. Training new model...')
        train_model()
