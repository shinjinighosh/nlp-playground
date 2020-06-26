import warnings
import json
import tqdm

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
urls = []

print("Loading from dataset...")
for news_report in tqdm.tqdm(dataset):
    sentences.append(news_report['headline'])  # ignoring the actual articles
    labels.append(news_report['is_sarcastic'])
    urls.append(news_report['article_link'])


# tokenizing sentences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print("Length of word index is ", len(word_index))
print(f"First sentence is \"{sentences[0]}\" and its representation is \n {padded[0]}")
print("Shape of padded sentences is ", padded.shape)
