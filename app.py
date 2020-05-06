from flask import Flask
from flask import jsonify
from flask_restful import reqparse, abort, Api, Resource
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import numpy as np
import pandas as pd     
from model import LinearSVCModel
from build_model import clean_recipe, le

app = Flask(__name__)
api = Api(app)

# create new model object
model = LinearSVCModel()

# load trained classifier
clf_path = 'lib/models/LinearSVC.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

# load trained vectorizer
vec_path = 'lib/models/tfidfVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

# prediction class, return in dict/JSON format

class PredictRecipe(Resource):
    def get(self):
        args = parser.parse_args()
        user_query = clean_recipe(args['query'])

        ## debug
        # print(user_query)
        # print(type(user_query))

        # vectorize the user's query and make a prediction
        uq_vec = model.vectorizer_transform(np.array([user_query]))
        prediction = model.predict(uq_vec).tolist()

        ## debug
        # print(prediction)
        # print(type(prediction))
        # print(le.inverse_transform(prediction).tolist()) 

        output = {'prediction': le.inverse_transform(prediction).tolist()}

        return jsonify(output)


# Setup the Api resource routing here, route the URL to the resource
api.add_resource(PredictRecipe, '/')

if __name__ == '__main__':
    app.run(debug=True)
