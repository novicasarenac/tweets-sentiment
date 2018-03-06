import pandas as pd
import matplotlib.pyplot as plt

from tweets_sentiment.preprocessing import transform_data as dt
from tweets_sentiment.preprocessing.constants import DATASET_DESTINATION
from tweets_sentiment.preprocessing.constants import PREPROCESSED_DATASET
from tweets_sentiment.preprocessing import clear_data as cd


def check_skewness():
    data = pd.read_csv(PREPROCESSED_DATASET)
    df = pd.DataFrame(data)
    print(df.groupby('sentiment').size())
    plt.hist(df.iloc[:, 1])
    plt.show()


def preprocess_data():
    data = pd.read_csv(DATASET_DESTINATION)
    preprocessed_data = pd.DataFrame(columns=['sentiment', 'tweet'])
    for index, row in data.iterrows():
        tweet = row['tweet']
        clean_tweet = cd.clear_data(tweet)
        tweet = cd.remove_special_characters(clean_tweet)
        tweet = dt.transform_post(tweet)
        preprocessed_data.loc[len(preprocessed_data)] = [row['sentiment'], tweet]
    preprocessed_data.to_csv(PREPROCESSED_DATASET)


if __name__ == '__main__':
    # preprocess_data()
    check_skewness()
