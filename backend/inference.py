
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from helper import get_sequences, REVERSE_LABEL_MAPPING
import pickle
import numpy as np


def load_model(path):
    return tf.keras.models.load_model(path)


app = FastAPI()


class Text(BaseModel):
    text: str


@app.post("/", response_model=Text)
def inference(user_request: Text):
    text = user_request.text
    model = load_model('saved_model/my_model')
    # loading
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    sequence = get_sequences([text], tokenizer)
    sentiment = np.argmax(model(sequence))
    return {"text": REVERSE_LABEL_MAPPING[sentiment]}


if __name__ == "__main__":
    uvicorn.run("inference:app", host="0.0.0.0", port=8000)
