# downloads needed nltk data
python -m nltk.downloader all

# downloads GloVe
python3 -m gensim.downloader --download glove-twitter-100
mv ~/gensim-data/glove-twitter-100/* tweets_sentiment/data/
gunzip tweets_sentiment/data/glove-twitter-100.gz
python3 setup.py install --user
