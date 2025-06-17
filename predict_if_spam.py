import numpy
import pandas
import pickle
import sklearn
from flask import Flask, render_template, request, jsonify
#import BERT_model
#print(dir(BERT_model))
from BERT_model import predict



app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
    #return "Welcome to the Sentiment Prediction API. Use POST /predict with JSON {'texts': [...]}."


@app.route("/predict", methods=["POST"])
def predict_sentiment():
    data = request.get_json()
    texts = data.get("texts")

    if not texts:
        return jsonify({"error" : "No inputs text provided"}), 400
    
    labels, confidences = predict(texts)

    results = []

    for text, label, prob in zip(texts, labels, confidences):
        sentiment = "Spam" if label == 1 else "Non-Spam"
        results.append({
            "text": text,
            "sentiment": sentiment,
            "confidence": float(prob[label])
        })

    return jsonify(results)
