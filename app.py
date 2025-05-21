from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model, tokenizer, and params
model = load_model("model_lstm_tuned.keras")
tokenizer = joblib.load("tokenizer_lstm.pkl")
params = joblib.load("lstm_params.pkl")

max_len = params["max_len"]
labels = ['negatif', 'netral', 'positif']

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)
    label = np.argmax(pred, axis=1)[0]
    return labels[label]

# Halaman utama dengan form HTML
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    text_sample = ""
    if request.method == "POST":
        text_sample = request.form["text"]
        sentiment = predict_sentiment(text_sample)
    return render_template("index.html", sentiment=sentiment, text_sample=text_sample)

# Endpoint GET API
@app.route("/api/generate", methods=["GET"])
def api_generate_get():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' query parameter"}), 400

    sentiment = predict_sentiment(text)
    return jsonify({
        "text": text,
        "sentiment": sentiment
    })

# Endpoint POST API
@app.route("/api/generate", methods=["POST"])
def api_generate_post():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON body"}), 400

    text = data["text"]
    sentiment = predict_sentiment(text)
    return jsonify({
        "text": text,
        "sentiment": sentiment
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host='0.0.0.0',port=port)
