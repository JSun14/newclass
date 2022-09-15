import streamlit as st
import requests

st.title("Sentiment Analysis Web App")
headline = st.text_input("Enter Your Financial Headline")
if st.button("Predict Sentiment"):
    res = requests.post(f"http://backend:8000/", json={"text": headline})
    result = res.json()
    st.write('The sentiment of this headline is ', result.get('text'))
