import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from gensim.models import KeyedVectors
from tweets_sentiment.preprocessing.constants import WORD2VEC_MODEL

WORDS_NUM = 5000


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
    print('===> Loading Word2Vec model...')
    word2vec_model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL,
                                                       binary=True)
    visualize_word2vec(word2vec_model, WORDS_NUM)
