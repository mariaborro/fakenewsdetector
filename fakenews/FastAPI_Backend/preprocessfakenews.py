import string
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
import spacy

def preprocess_title(title):

    #removing excess space
    title = title.strip()

    #making sure title is lowercase
    title = title.lower()

    #removing digits
    title = "".join(char for char in title if not char.isdigit())

    #removing punctuation and symbols
    for punctuation in string.punctuation:
        title = title.replace(punctuation,"")

    #tokenizing
    title = title.split()

    #removing stopwords
    sp = spacy.load(‘en_core_web_sm’)            
    stop_words = sp.Defaults.stop_words

    title = [w for w in title if not w in stop_words]

    #lemmatizing verds and nouns
    #title = [WordNetLemmatizer().lemmatize(w, pos = "v") for w in title]
    #title = [WordNetLemmatizer().lemmatize(w, pos = "n") for w in title]

    #converting tokens to string again
    #title = " ".join(title)

    return title
