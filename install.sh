downloads needed nltk data
python -m nltk.downloader all

download GloVe
python3 -m gensim.downloader --download glove-twitter-100
mv ~/gensim-data/glove-twitter-100/* tweets_sentiment/data/
gunzip tweets_sentiment/data/glove-twitter-100.gz

download Word2Vec
python3 -m gensim.downloader --download word2vec-google-news-300
mv ~/gensim-data/word2vec-google-news-300/* tweets_sentiment/data/
gunzip tweets_sentiment/data/word2vec-google-news-300.gz

python3 setup.py install --user
