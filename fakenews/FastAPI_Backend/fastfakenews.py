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
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer

app = FastAPI()

#this is the loading we would do executing locally:
#app.state.model = joblib.load('model_for_deployment')
#this is the loading we do for production:
#app.state.model = joblib.load('fakenews/FastAPI_Backend/model_for_deployment')

#app.state.labels = {0: 'fake',1: 'true'}

@app.get("/")
def home():
    return {"Message": "Welcome to our Fake News Predictor!"}

#as a reminder now we can run this code in our command line: uvicorn fastfakenews:app --reload


@app.get('/predict')
async def make_prediction(text):
    """Returns the prediction of a pasted text
    """
    new_title = text.strip()
    new_title = new_title.lower()
    new_title = "".join(char for char in new_title if not char.isdigit())
    for punctuation in string.punctuation:
        new_title = new_title.replace(punctuation,"")
    #new_title = word_tokenize(new_title)
    #stop_words = set(stopwords.words('english'))
    #new_title = [w for w in new_title if not w in stop_words]
    #new_title = [WordNetLemmatizer().lemmatize(w, pos = "v") for w in new_title]
    #new_title = [WordNetLemmatizer().lemmatize(w, pos = "n") for w in new_title]
    new_title = " ".join(new_title)
    return {"prova": new_title}
    #tk = load_tokenizer()
    #title_preprocessed_token = tk.texts_to_sequences([title_preprocessed])
    #prediction =  app.state.model.predict(title_preprocessed_token)
    #numerot = 0 if prediction[0] < 0.5 else 1
    #predicted_category = app.state.labels.get( numerot )
    #return {"prediction": predicted_category}
