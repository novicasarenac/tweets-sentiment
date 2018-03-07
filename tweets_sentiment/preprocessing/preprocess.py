import enchant
import pandas as pd
import matplotlib.pyplot as plt

from tweets_sentiment.preprocessing.clear_data import clear_data
from tweets_sentiment.preprocessing.transform_data import transform_post
from tweets_sentiment.preprocessing.transform_data import load_sleng_dict
from tweets_sentiment.preprocessing.transform_data import init_tokenizer
from tweets_sentiment.preprocessing.constants import DATASET_DESTINATION
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET


def check_skewness():
    data = pd.read_csv(PREPROCESSED_DATASET)
    df = pd.DataFrame(data)
    print(df.groupby('sentiment').size())
    plt.hist(df.iloc[:, 1])
    plt.show()


def preprocess_and_save(column, preprocess_data, tokenizer, slang_dict, checker):
    print(column[0])
    tweet = column[2]
    clean_tweet = clear_data(tweet)
    tweet = transform_post(clean_tweet, tokenizer, slang_dict, checker)
    preprocess_data.loc[len(preprocess_data)] = [column[1], tweet]


def preprocess_data():
    data = pd.read_csv(DATASET_DESTINATION)
    tokenizer = init_tokenizer(strip_handles=True)
    slang_dictionary = load_sleng_dict()
    checker = enchant.Dict('en_US')
    preprocessed_data = pd.DataFrame(columns=['sentiment', 'tweet'])
    print("===> Start preprocessing dataset...")
    data.apply(lambda col: preprocess_and_save(col, preprocessed_data, tokenizer, slang_dictionary, checker), axis=1)
    preprocessed_data.to_csv(PREPROCESSED_DATASET)
    print("===> Preprocessing finished...")


if __name__ == '__main__':
    preprocess_data()
    # check_skewness()
