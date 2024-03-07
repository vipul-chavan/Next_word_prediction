import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import numpy as np
import streamlit as st

# Load the pre-trained model and tokenizer
model = load_model('D:\\BE_IT\\te\\dsbda\\Next_Word_Prediction_V2\\next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    return predicted_word

# Streamlit UI
st.title("Next Word Prediction")

# User input text box
user_input = st.text_input("Enter your text:")

if user_input:
    try:
        # Tokenize the last 3 words from the user input
        user_input_tokens = user_input.split(" ")
        user_input_tokens = user_input_tokens[-3:]

        # Make prediction using the model
        prediction = Predict_Next_Words(model, tokenizer, user_input_tokens)
        
        # Display the result
        st.success(f"Predicted next word: {prediction}")
    except Exception as e:
        st.error(f"Error occurred: {e}")
