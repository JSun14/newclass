import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import tensorflow as tf

data = pd.read_csv('all-data.csv',
                   encoding='latin-1', names=['Label', 'Text'])

# Preprocessing


def get_sequences(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    print("Vocab length:", len(tokenizer.word_index) + 1)

    max_seq_length = np.max(list(map(lambda x: len(x), sequences)))
    print("Maximum sequence length:", max_seq_length)

    sequences = pad_sequences(sequences, maxlen=max_seq_length, padding='post')

    return sequences


def preprocess_inputs(df):
    df = df.copy()

    sequences = get_sequences(df['Text'])

    label_mapping = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }

    y = df['Label'].replace(label_mapping)

    train_sequences, test_sequences, y_train, y_test = train_test_split(
        sequences, y, train_size=0.7, shuffle=True, random_state=1)

    return train_sequences, test_sequences, y_train, y_test


train_sequences, test_sequences, y_train, y_test = preprocess_inputs(data)

inputs = tf.keras.Input(shape=(train_sequences.shape[1],))

x = tf.keras.layers.Embedding(
    input_dim=10123,
    output_dim=128,
    input_length=train_sequences.shape[1]
)(inputs)
x = tf.keras.layers.GRU(256, return_sequences=True, activation='tanh')(x)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_sequences,
    y_train,
    validation_split=0.2,
    batch_size=32,
    epochs=10,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

results = model.evaluate(test_sequences, y_test, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
