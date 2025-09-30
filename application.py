from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## Import ridge regressor, standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl', 'rb')) # Read byte mode(stream)
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        # For new data means test data only do transform data dont do fit_transform because that is for training data
        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled) # RESsult will be in form of a list and contains only one value

        return render_template('home.html', results = result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host = "0.0.0.0") # This will map to the local ip address of any machine you are working(So no problem with running in different machine)
# By default flask runs on port number 5000 if you want to change that in app.run(host = "0.0.0.0", port = x) here we can define our own ip address(x)

# 192.168.1.2 is the ip address where this application is running locally in ur system