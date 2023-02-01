from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from preprocessfakenews import preprocess_title
from sklearn.model_selection import train_test_split

def load_train_data():
    og_data = pd.read_csv("raw_data/dataset.csv")
    data = og_data.dropna()
    data.reset_index(inplace=True)
    data = data[["title", "text", "label"]]
    X = data.drop(columns = ["label"])
    y = data["label"]
    X_title = X.drop("text", axis=1)
    X_title_preprocessed = [preprocess_title(title) for title in X_title.title]
    X_title_train, X_title_test, y_title_train, y_title_test = train_test_split(X_title_preprocessed, y)
    return X_title_train

def load_tokenizer():
    tk = Tokenizer()
    X_title_train = load_train_data()
    tk.fit_on_texts(X_title_train)
    return tk
