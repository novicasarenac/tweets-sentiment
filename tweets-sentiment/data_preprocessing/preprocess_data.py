import csv

path = '../../data/data.csv'

def main():
    with open(path, 'rb') as csvfile:
        data = [next(csvfile) for x in range(100)]
    for x in data:
        print(bytes.decode(x))

if __name__ == "__main__":
    main()
