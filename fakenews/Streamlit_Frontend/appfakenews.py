import streamlit as st
import requests
#from PIL import Image

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


st.title('Welcome to our Fake News Classificator!')

st.write("Now you can check whether what you are about to read is fake news or not!!")

st.info("made with a LSTM model")

#image = Image.open('fakenewsimage.jpg')

#st.image(image)

st.subheader("Input the News title below")

sentence = st.text_area("Title:", "Example: Cats are undercover CIA agents",height=200)

predict_btt = st.button("PREDICT")


#this is the url we would use if running locally:
#url = "http://127.0.0.1:8000/predict"
#this is the url we use for deploying:
url = "https://api-fakenews.onrender.com/predict"

if predict_btt:
    with st.spinner("Please wait :)"):
        with requests.Session() as session:
            response = session.get(url, params= {"text": sentence})
            result = response.json()["prediction"]
            st.markdown(f"These news are: {result}!!")
            st.balloons()
            print(response.status_code)
            print(response.content)
            print(response.json())

 # as a reminder, after activating uvicorn with uvicorn fastfakenews:app --reload we can run in our command line: streamlit run appfakenews.py
