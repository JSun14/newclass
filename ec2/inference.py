
import streamlit as st
import tensorflow as tf
from helper import get_sequences, REVERSE_LABEL_MAPPING
import pickle
import numpy as np


def load_model(path):
    return tf.keras.models.load_model(path)


model = load_model('saved_model/my_model')
st.title("Sentiment Analysis Web App")
headline = st.text_input("Enter Your Financial Headline")
if st.button("Predict Sentiment"):
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequence = get_sequences([headline], tokenizer)
    sentiment = np.argmax(model(sequence))
    result = REVERSE_LABEL_MAPPING[sentiment]
    st.write('The sentiment of this headline is ', result)
