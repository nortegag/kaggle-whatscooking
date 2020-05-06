import pandas as pd
import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

class LinearSVCModel(object):
    def __init__(self):
        """Initialize Linear SVC and tf-idf vectorizer
        Attributes:
            clf: sklearn classifier model
            vectorizer: tfidf vectorizer (or BoW)
        """
        self.clf = LinearSVC(random_state=3, 
                             max_iter=10000)
        self.vectorizer = TfidfVectorizer()

    def vectorizer_fit(self, X):
        """Fit data to vectorizer
        """
        self.vectorizer.fit(X)
        

    def vectorizer_transform(self, X):
        """Transform data to sparse matrix
        """
        X_trans = self.vectorizer.transform(X)
        return X_trans

    def train(self, X, y):
        """Trains classifier for label/cuisine prediction 
        """
        self.clf.fit(X, y)

    def predict(self, X):
        le = LabelEncoder()
        y_pred = self.clf.predict(X)
        return y_pred
        
    def pickle_clf(self, path="lib/models/LinearSVC.pkl"):
        """Saves trained Linear SVC classifier 
        """
        with open(path, "wb") as f:
            pickle.dump(self.clf, f)
            print("Pickled classifier: {}".format(path))
    
    def pickle_vectorizer(self, path="lib/models/tfidfVectorizer.pkl"):
        """Saves trained TF-IDF vectorizer 
        """
        with open(path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
            print("Pickled vectorizer: {}".format(path))


    

    

    