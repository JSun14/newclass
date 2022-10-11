import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
from helper import preprocess_inputs
import wandb
from wandb.keras import WandbCallback

data = pd.read_csv('all-data.csv',
                   encoding='latin-1', names=['Label', 'Text'])

# Preprocessing

train_sequences, test_sequences, y_train, y_test = preprocess_inputs(data)

inputs = tf.keras.Input(shape=(train_sequences.shape[1],))

lr = 0.001
epochs = 5
batch_size = 32
beta_1 = 0.9
beta_2 = 0.999
loss = 'sparse_categorical_crossentropy'
config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": batch_size,
    "loss": loss,
    "beta_1": beta_1,
    "beta_2": beta_2
}
wandb.init(project="test-project",
           entity="cornell-data-science-class", config=config)

x = tf.keras.layers.Embedding(
    input_dim=10123,
    output_dim=128,
    input_length=train_sequences.shape[1]
)(inputs)
x = tf.keras.layers.GRU(128, return_sequences=True, activation='tanh')(x)
x = tf.keras.layers.Flatten()(x)
outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
opt = tf.keras.optimizers.Adam(learning_rate=lr)

model.compile(
    optimizer=opt,
    loss=loss,
    metrics=['accuracy']
)

history = model.fit(
    train_sequences,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[
        WandbCallback()
    ]
)

results = model.evaluate(test_sequences, y_test, verbose=0)
print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))
