import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# model loading
model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)[0]  # Extracting the prediction from the array
    prediction = round(prediction, 2)  # Rounding the prediction to 2 decimal places

    return render_template("index.html", prediction_text="Price Prediction is {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
