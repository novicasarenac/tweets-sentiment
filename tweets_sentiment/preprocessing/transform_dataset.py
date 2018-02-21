import pandas as pd
from os import path


FULL_PATH = path.dirname(path.abspath(__file__ + "/../"))
FULL_DATASET = path.join(FULL_PATH, 'data/text_emotion.csv')
DATASET_DESTINATION = path.join(FULL_PATH, 'data/new_data.csv')


SENTIMENTS = {
    'positive': [
        'love',
        'surprise',
        'fun',
        'relief'
        'happiness',
        'enthusiasm',
    ],
    'negative': [
        'sadness',
        'worry',
        'hate',
        'boredom',
        'anger'
    ],
    'unlabelled': [
        'empty',
        'neutral'
    ]
}


def transform_dataset():
    data = pd.read_csv(FULL_DATASET)
    new_data = pd.DataFrame(columns=['sentiment', 'tweet'])

    for index, tweet in data.iterrows():
        if tweet['sentiment'] not in SENTIMENTS['unlabelled']:
            new_data.loc[len(new_data)] = [0 if tweet['sentiment'] in SENTIMENTS['negative'] else 1, tweet['content']]
    new_data.to_csv(DATASET_DESTINATION)


if __name__ == '__main__':
    transform_dataset()
