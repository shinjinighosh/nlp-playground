import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer


sentences = ["My name is Shinjini",
             "My plushie's name is Bruno",
             "My friend loves my plushie",
             "I love it too!",
             "My plushie is very cute"]

# tokenize sentences
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print('The word index is ', word_index)

# generate sequences out of tokens
sequences = tokenizer.texts_to_sequences(sentences)
print('Training data sequences are', sequences)


# testing
test_data = ["But I really love my plushie",
             "My friend wants a plushie too!"]

test_seq = tokenizer.texts_to_sequences(test_data)
print('Testing data sequences are ', test_seq)
