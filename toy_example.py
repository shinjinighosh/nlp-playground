import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences


sentences = ["My name is Shinjini",
             "My plushie's name is Bruno",
             "My friend loves my plushie",
             "I love it too!",
             "My plushie is very cute"]

# tokenize sentences
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print('The word index is\n', word_index)

# generate sequences out of tokens
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=10)
print('Training data sequences are\n', sequences)
print('Training data padded sequences are\n', padded)


# testing
test_data = ["But I really love my plushie",
             "My friend wants a plushie too!"]

test_seq = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_seq, maxlen=10)
print('Testing data sequences are\n', test_seq)
print('Testing data padded sequences are\n', test_padded)
