import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model("models/chatbot_model.h5")
tokenizer = pickle.load(open("models/tokenizer.pkl", "rb"))
lbl_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
max_len = 20

def predict_class(text, model):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len, truncating='post')
    res = model.predict(padded)[0]
    return lbl_encoder.inverse_transform([np.argmax(res)])[0]

def get_response(tag, intents):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])
    return "I am not sure how to respond to that."

st.title("Mental Health Chatbot")
user_input = st.text_input("You: ")
if st.button("Send"):
    import json
    with open("data/intents.json") as file:
        intents = json.load(file)
    predicted_class = predict_class(user_input, model)
    response = get_response(predicted_class, intents)
    st.text_area("Chatbot:", value=response, height=150)
