import warnings
import numpy as np
from tqdm import tqdm

print("Ignoring Tensorflow warnings")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    import tensorflow_datasets as tfds
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences


# print(tf.__version__)
tf.enable_eager_execution()

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
    training_labels.append(str(label.numpy()))

print("Generating testing data")
for sentence, label in tqdm(testing_data):
    testing_sentences.append(str(sentence.numpy()))
    testing_labels.append(str(label.numpy()))

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)


# tokenizing
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

testing_sequences = tokenizer.texts_to_sequences(testing_sequences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)
