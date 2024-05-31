import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open("final_pipe_model", 'rb'))
@app.route("/")
def Home():
    return render_template("index.html")
@app.route("/predict", methods = ["POST"])
def predict():
    species_mapping = ['Setosa', 'Versicolor', 'Virginica']
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = species_mapping[int(model.predict(features))]
    return render_template("index.html", prediction_text = "The flower type is {}".format(prediction))
if __name__ == "__main__":
    app.run(debug=True)
