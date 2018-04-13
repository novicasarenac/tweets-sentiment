import enchant
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt
import dask.dataframe as dd

from dask.multiprocessing import get
from tweets_sentiment.preprocessing.clear_data import clear_data
from tweets_sentiment.preprocessing.transform_data import transform_post
from tweets_sentiment.preprocessing.transform_data import load_sleng_dict
from tweets_sentiment.preprocessing.transform_data import init_tokenizer
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET
from tweets_sentiment.preprocessing.constants import LARGE_DATASET_RAW

CPU_CORES = multiprocessing.cpu_count()


def read_corpus_dataset(dataset_path):
    data = pd.read_csv(dataset_path, error_bad_lines=False)
    labels = data['Sentiment']
    tweets = data['SentimentText']

    return tweets, labels


def check_skewness(dataset_path):
    data = pd.read_csv(dataset_path, error_bad_lines=False)
    df = pd.DataFrame(data)
    df.groupby('Sentiment').size().plot(kind='bar')
    plt.show()


def preprocess_and_save(tweet, tokenizer, slang_dict, checker):
    clean_tweet = clear_data(tweet)
    return transform_post(clean_tweet, tokenizer, slang_dict, checker)


def preprocess_data(dataset_path):
    data = pd.read_csv(dataset_path, error_bad_lines=False)
    tokenizer = init_tokenizer(strip_handles=True)
    slang_dictionary = load_sleng_dict()
    checker = enchant.Dict('en_US')
    preprocess_data = pd.DataFrame(columns=['Sentiment', 'SentimentText'])
    dask_df = dd.from_pandas(data, npartitions=CPU_CORES)
    print("===> Start preprocessing dataset...")
    processed_data = dask_df.map_partitions(
            lambda partition: partition.apply(
                lambda col: preprocess_and_save(col[3],
                                                tokenizer,
                                                slang_dictionary,
                                                checker), axis=1)
            ).compute(get=get)
    preprocess_data.iloc[:, 0] = data.iloc[:, 1].values
    preprocess_data.iloc[:, 1] = processed_data.values
    preprocess_data.to_csv(PREPROCESSED_DATASET)
    print("===> Preprocessing finished...")


if __name__ == '__main__':
    preprocess_data(LARGE_DATASET_RAW)
    check_skewness(LARGE_DATASET_RAW)
