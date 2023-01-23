import streamlit as st
import requests


st.title('Welcome to our Fake News Classificator!')
st.write("You can check here if what you are about to read is fake news or not!!")
st.info("LSTM model")
st.subheader("Input the News title below")
sentence = st.text_area("Enter your news title here", "Example: Cats are CIA agents",height=200)
predict_btt = st.button("predict")
url = "http://127.0.0.1:8000/predict"
if predict_btt:
    with requests.Session() as session:
        response = session.get(url, params= {"text": predict_btt})
        result = response.json()["prediction"]
        st.markdown(f"These news are: {result}!!")
        print(response.status_code)
        print(response.content)
        print(response.json())
