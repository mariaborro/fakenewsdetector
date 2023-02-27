import string
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#from nltk.stem import WordNetLemmatizer
#import spacy
#from textblob import TextBlob, Word
#import pattern
#from pattern.en import lemma, lexeme
#from gensim.utils import lemmatize
import treetaggerwrapper as ttpw

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
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

    title = [w for w in title if not w in stop_words]

    #converting tokens to string again
    title = " ".join(title)

    #lemmatizing:
    tagger = ttpw.TreeTagger(TAGLANG='en', TAGDIR='/Users/ecom-selva.p/Documents/MLPlus/11_Lemmatization/treetagger')
    tags = tagger.tag_text(title)
    title = [t.split('\t')[-1] for t in tags]

    return title
