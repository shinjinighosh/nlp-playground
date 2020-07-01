#!/usr/bin/python3

import warnings
import json
import tqdm
import matplotlib.pyplot as plt

print("Ignoring Tensorflow warnings")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

# loading dataset
with open("Data/sarcasm.json", "r") as f:
    dataset = json.load(f)

sentences = []
labels = []
# urls = []

print("Loading from dataset...")
for news_report in tqdm.tqdm(dataset):
    sentences.append(news_report['headline'])  # ignoring the actual articles
    labels.append(news_report['is_sarcastic'])
    # urls.append(news_report['article_link'])

# tokenizing sentences
vocab_size = 10000
embedding_dim = 16
max_length = 32
trucation_type = 'post'
padding_type = 'post'
oov_token = "<OOV>"
training_size = 20000

# train test split
training_sentences = sentences[:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[:training_size]
testing_labels = labels[training_size:]

print("Tokenizing")
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

print("Creating training sequences")
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length,
                                padding=padding_type, truncating=trucation_type)

print("Creating testing sequences")
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length,
                               padding=padding_type, truncating=trucation_type)

# print("Length of word index is ", len(word_index))
# print(f"First sentence is \"{sentences[0]}\" and its representation is \n {padded[0]}")
# print("Shape of padded sentences is ", padded.shape)

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
history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels))

# plotting model
print("Creating plots")


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    # plt.savefig(string + '_sarcasm.jpg')
    plt.show()


# print(history.history.keys())
plot_graphs(history, "acc")
plot_graphs(history, "loss")
