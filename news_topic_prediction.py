#!/usr/bin/python3

import warnings
import tqdm
import csv
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt


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

# train test split
training_size = int(0.7 * len(sentences))
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

# tokenizing content
vocab_size = 10000
max_length = 32
trucation_type = 'post'
padding_type = 'post'
embedding_dim = 16

print("Tokenizing")
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
# sequences = tokenizer.texts_to_sequences(sentences)
# padded = pad_sequences(sequences, padding='post')

print("Creating training sequences")
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                padding=padding_type, truncating=trucation_type)

print("Creating testing sequences")
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                               padding=padding_type, truncating=trucation_type)


# tokenizing labels
print("Tokenizing labels...")
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(training_labels)
training_label_padded = pad_sequences(label_seq, padding='post')
testing_label_seq = label_tokenizer.texts_to_sequences(testing_labels)
testing_label_padded = pad_sequences(testing_label_seq, padding='post')

print("Length of word index is ", len(word_index))
# print(f"Representation of the first sentence is \n {training_padded[0]}")
# print("Shape of padded sentences is ", training_padded.shape)
print("Label word index is ", label_word_index)


# creating the model
print("Building the model")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(24, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# training
print("Starting training")

num_epochs = 10
history = model.fit(training_padded, training_label_padded, epochs=num_epochs,
                    validation_data=(testing_padded, testing_label_padded))

# plotting model
print("Creating plots")


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    # plt.savefig(string + '_news_topic.jpg')
    plt.show()


# print(history.history.keys())
plot_graphs(history, "acc")
plot_graphs(history, "loss")
