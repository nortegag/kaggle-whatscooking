import pandas as pd
import os
import re
from model import LinearSVCModel
from sklearn.preprocessing import LabelEncoder
from nltk import wordpunct_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from datetime import datetime

def clean_recipe(x):
    """Clean our ingredients list into stemmed and lemmatized string of tokens
    Input: list of ingredients (or string, for API)
    """
    #stemmer and lemmatizer objects from nltk
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    prePro = ""
    if type(x) == type(list()):
        x = " ".join(x)
    x = x.lower()
    x = re.sub("[-'%]", " ", x)
    x = re.sub("[^0-9a-zA-Z ]+", "", x)
    #removes non-ASCII characters
    x = re.sub("[^\x00-\x7f]", "", x) 
    #tokenization
    words = wordpunct_tokenize(x)
    words_out = []
    #stem and lemmatize
    for word in words:
        word = lemmatizer.lemmatize(stemmer.stem(word))
        words_out.append(word)
    prePro = ' '.join(words_out)    
        
    return prePro

#getting our train/test data, preprocessed with trained label encoder
with open("{}/lib/data/train.json".format(os.getcwd())) as f:
    train_data = pd.read_json(f)

# create new feature with preprocessed recipe for model training
train_data["prePro"] = train_data.ingredients.apply(lambda x: clean_recipe(x))

# separate into labels/targets
X_train = train_data.prePro
y_train = train_data.cuisine

#fitting label encoder for cuisine prediction
le = LabelEncoder()
y = le.fit_transform(y_train.values)


def build_model():
    """Build our model with training data from Kaggle
    """
    startTime = datetime.now()
    model = LinearSVCModel() #scitkit-learn LSVC model
    currpath = os.getcwd() #current directory

    #fitting data into our vectorizer
    model.vectorizer_fit(X_train.values) 
    print("TFIDF fit complete", datetime.now() - startTime)

    X = model.vectorizer_transform(X_train.values)
    print("TFIDF transform complete", datetime.now() - startTime)

    #train the model with X and y generated
    model.train(X, y)
    print("Model training successful", datetime.now() - startTime)

    #serializing/pickling the trained classifier and vectorizer 
    model.pickle_clf()
    model.pickle_vectorizer()

    print(datetime.now() - startTime) 

if __name__ == "__main__":
    build_model()