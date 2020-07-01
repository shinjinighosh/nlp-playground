#!/usr/bin/python3

import warnings
import numpy as np
from tqdm import tqdm
import io

print("Ignoring Tensorflow warnings")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    import tensorflow_datasets as tfds
    # from tensorflow import keras
    # from tensorflow.keras import layers
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences


# print(tf.__version__)
tf.compat.v1.enable_eager_execution()

# loading dataset
print("Loading dataset...")
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
training_data, testing_data = imdb['train'], imdb['test']

training_sentences = []
testing_sentences = []
training_labels = []
testing_labels = []

print("Generating training data")
for sentence, label in tqdm(training_data):
    training_sentences.append(str(sentence.numpy()))
    training_labels.append(label.numpy())

print("Generating testing data")
for sentence, label in tqdm(testing_data):
    testing_sentences.append(str(sentence.numpy()))
    testing_labels.append(label.numpy())

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)


# tokenizing
print("Tokenizing")

vocab_size = 10000
embedding_dim = 16
max_length = 150
trucation_type = 'post'
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trucation_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)  # truncate them too?

# creating the model
print("Building the model")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
# model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(6, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# training
print("Starting training")

num_epochs = 10
history = model.fit(padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels))

# inspect
embedding_layer = model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]
print(embedding_weights.shape)  # (vocab_size, embedding_dim)

reverse_word_index = dict([(value, key) for key, value in word_index.items()])

out_v = io.open('vecs_imdb.tsv', 'w', encoding='utf-8')
out_m = io.open('meta_imdb.tsv', 'w', encoding='utf-8')

for word_num in range(1, vocab_size):  # ignoring <OOV>
    word = reverse_word_index[word_num]
    embeddings = embedding_weights[word_num]
    out_m.write(word + '\n')
    out_v.write('\t'.join([str(e) for e in embeddings]) + "\n")

out_v.close()
out_m.close()
