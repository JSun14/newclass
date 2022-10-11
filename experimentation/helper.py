import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import pickle

MAX_SEQ_LENGTH = 71
LABEL_MAPPING = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def get_sequences(texts, tokenizer=None):
    if tokenizer == None:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    print("Vocab length:", len(tokenizer.word_index) + 1)

    # MAX_SEQ_LENGTH = np.max(list(map(lambda x: len(x), sequences)))
    print("Maximum sequence length:", MAX_SEQ_LENGTH)
    sequences = pad_sequences(
        sequences, maxlen=MAX_SEQ_LENGTH, padding='post')
    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle)

    return sequences


def preprocess_inputs(df):
    df = df.copy()

    sequences = get_sequences(df['Text'])

    y = df['Label'].replace(LABEL_MAPPING)

    train_sequences, test_sequences, y_train, y_test = train_test_split(
        sequences, y, train_size=0.7, shuffle=True, random_state=1)

    return train_sequences, test_sequences, y_train, y_test
