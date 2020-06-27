import warnings
import numpy as np
from tqdm import tqdm

print("Ignoring Tensorflow warnings")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    import tensorflow_datasets as tfds


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
