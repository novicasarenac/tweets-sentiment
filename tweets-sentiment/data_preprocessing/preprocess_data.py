import csv
import clear_data as cd
import data_transform as dt

path = '../../data/data.csv'

processedData = '../../data/preprocessedData.csv'
separator = 'Sentiment140,'

def preprocessing(data):
    with open(processedData, "w") as preprocessedData_file:

        for index, x in enumerate(data):
            if separator in x:
                sentiment = x.split(',')[1]
                tweet = x.split(separator)[1]
                #TODO: pozvati funkcije, proslediti tweet
                tweet =  dt.transform_post(cd.clear_data(tweet))
                tweet = cd.remove_special_characters(tweet)
                line = sentiment + separator + tweet + " ==== " + x + "\n"
                preprocessedData_file.write(line)

    preprocessedData_file.close()

def main():
    with open(path, 'r') as csvfile:
        data = [next(csvfile) for x in range(100)]
    preprocessing(data)

if __name__ == "__main__":
    main()







