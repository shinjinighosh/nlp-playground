#!/usr/bin/python3

import warnings
import tqdm
import csv
from nltk.tokenize import word_tokenize


print("Ignoring Tensorflow warnings")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# loading data and removing stopwords
stopwords = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
                 "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"])
print("Loading dataset...")
sentences = []
labels = []

with open("Data/bbc-text.csv", "r") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm.tqdm(reader):
        labels.append(row['category'])
        original_sentence = row['text']
        text_tokens = word_tokenize(original_sentence)
        pruned_sentence = [word for word in text_tokens if word not in stopwords]
        sentences.append(" ".join(pruned_sentence))

print("=========================")
print(f"There are {len(sentences)} sentences")
# print(f"The first sentence is\n{sentences[0]}")

# tokenizing content
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')

# tokenizing labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)
label_padded = pad_sequences(label_seq, padding='post')

print("Length of word index is ", len(word_index))
print(f"Representation of the first sentence is \n {padded[0]}")
print("Shape of padded sentences is ", padded.shape)
print("Label word index is ", label_word_index)
