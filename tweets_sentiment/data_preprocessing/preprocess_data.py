import clear_data as cd
import data_transform as dt

path = '../../data/data.csv'

testData = '../../data/testData.csv'
processedData = '../../data/preprocessedData.csv'
separator = 'Sentiment140,'


def vector_representation(fileName):
    with open(fileName, 'r') as testFile:
        data = [next(testFile) for x in range(19966)]

    sentiments = []
    tweets = []
    for index, x in enumerate(data):
        sentiment, tweet = x.split(separator)
        sentiments.insert(index, sentiment)
        tweets.insert(index, tweet)

    return list(map(int, sentiments)), tweets


def preprocessing(data, fileName):
    with open(fileName, "w") as preprocessedData_file:
        for index, x in enumerate(data):
            if separator in x:
                sentiment = x.split(',')[1]
                tweet = x.split(separator)[1]
                #TODO: pozvati funkcije, proslediti tweet
                tweet = dt.transform_post(cd.clear_data(tweet))
                tweet = cd.remove_special_characters(tweet)
                line = sentiment + separator + tweet + "\n"
                preprocessedData_file.write(line)

        preprocessedData_file.close()


def main():
   # with open(path, 'r', encoding="utf8") as csvfile:
   #     data = [next(csvfile) for x in range(35000)]
   # preprocessing(data[0:20000], testData)
   # preprocessing(data[20000:35000], processedData)
    vector_representation(testData)


if __name__ == "__main__":
    main()