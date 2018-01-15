import csv

path = '../../data/data.csv'

def main():
    with open(path, 'rb') as csvfile:
        data = [next(csvfile) for x in xrange(100)]
    for x in data:
        print(x + "\n")

if __name__ == "__main__":
    main()
