import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
import numpy as np



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
    print(predicted_word)
    return predicted_word


st.title("Next Word predication ")

user_input = st.text_input("Enter your line :")

if user_input is not None :
    try:
        user_input = user_input.split(" ")
        user_input = user_input[-3:]

        prediction = Predict_Next_Words(model, tokenizer, user_input)
        #st.success(f"Predicted next word: {prediction}")
        st.write(f"Predicted next word :{prediction}")
    except Exception as e:
        st.error(f"Error occurred: {e}")
