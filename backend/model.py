import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from backend.helper import preprocess_inputs

data = pd.read_csv('./storage/all-data.csv',
                   encoding='latin-1', names=['Label', 'Text'])

# Preprocessing

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
    epochs=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
    ]
)

results = model.evaluate(test_sequences, y_test, verbose=0)
if not os.path.exists('saved_model'):
    os.mkdir('saved_model')
model.save('saved_model/my_model')
print(type(test_sequences[0]))
print(
    model(np.array([test_sequences[0]])))
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
