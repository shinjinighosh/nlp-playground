import warnings
import json

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("Data/sarcasm.json", "r") as f:
    dataset = json.load(f)

sentences = []
labels = []
urls = []

for news_report in dataset:
    sentences.append(news_report['headline'])  # ignoring the actual articles
    labels.append(news_report['is_sarcastic'])
    urls.append(news_report['article_link'])
