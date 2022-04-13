# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 22:00:13 2022

@author: User
"""
from flask import Flask, render_template, request
import audio_classification

app = Flask(__name__)


@app.route('/')
def hello():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    audio_file=request.files['file']
    prediction=audio_classification.audio_classification(audio_file)

    return render_template("index.html",prediction_text="Class of audio file {}".format(prediction))


if __name__ == "__main__":
    app.run(debug=True)

