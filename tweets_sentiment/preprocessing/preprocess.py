from os import path
import pandas as pd
from tweets_sentiment.data_preprocessing import clear_data as cd
from tweets_sentiment.data_preprocessing import data_transform as dt


FULL_PATH = path.dirname(path.abspath(__file__ + "/../"))
DATASET_DESTINATION = path.join(FULL_PATH, 'data/new_data.csv')
PREPROCESSED_DATASET = path.join(FULL_PATH, 'data/preprocessed_dataset.csv')


def preprocess_data():
    data = pd.read_csv(DATASET_DESTINATION)
    preprocessed_data = pd.DataFrame(columns = ['sentiment', 'tweet'])
    for index, row in data.iterrows():
        tweet = row['tweet']
        tweet = dt.transform_post(cd.clear_data(tweet))
        tweet = cd.remove_special_characters(tweet)
        preprocessed_data.loc[len(preprocessed_data)] = [row['sentiment'], tweet]
    preprocessed_data.to_csv(PREPROCESSED_DATASET)


if __name__ == '__main__':
    preprocess_data()
