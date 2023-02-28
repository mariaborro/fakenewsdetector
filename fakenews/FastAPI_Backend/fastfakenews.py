import nltk
nltk.download('wordnet')
from fastapi import FastAPI
#from tensorflow.keras.models import load_model
#this is the import for locally executing the api:
#from preprocessfakenews import preprocess_title
#this is the import for deploying the api:
from fakenews.FastAPI_Backend.preprocessfakenews import preprocess_title
from tensorflow.keras.preprocessing.sequence import pad_sequences
#this is the import for locally executing the api:
#from tokenizercreator import load_tokenizer
#this is the import for deploying the api:
from fakenews.FastAPI_Backend.tokenizercreator import load_tokenizer
import joblib
import numpy as np
import string
#import subprocess
#cmd = ["python3", "-m", "nltk.downloader", "all"]
#subprocess.run(cmd)
#from fakenews.FastAPI_Backend.nltk_download_utils import dothething
#dothething()
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

app = FastAPI()

#this is the loading we would do executing locally:
#app.state.model = joblib.load('model_for_deployment')
#this is the loading we do for production:
app.state.model = joblib.load('fakenews/FastAPI_Backend/model_for_deployment')

app.state.labels = {0: 'true',1: 'fake'}

@app.get("/")
def home():
    return {"Message": "Welcome to our Fake News Predictor!"}

#as a reminder now we can run this code in our command line: uvicorn fastfakenews:app --reload


@app.get('/predict')
async def make_prediction(text):
    """Returns the prediction of a pasted text
    """
    new_title = preprocess_title(text)
    
    tk = load_tokenizer()

    title_preprocessed_token = tk.texts_to_sequences([new_title])

    prediction =  app.state.model.predict(title_preprocessed_token)

    numerot = 0 if prediction[0] < 0.5 else 1

    predicted_category = app.state.labels.get( numerot )

    return {"prediction": predicted_category}
