from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__, template_folder="../templates")

# Load dataset
df = pd.read_csv("mail_data.csv")

# Encode spam / ham
encoder = LabelEncoder()
df['Category'] = encoder.fit_transform(df['Category'])

x = df['Message']
y = df['Category']

# Text vectorization
vectorizer = TfidfVectorizer(stop_words='english')
x_vectorized = vectorizer.fit_transform(x)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_vectorized, y)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    message = data["message"]

    vector = vectorizer.transform([message])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "Spam Email ❌"
    else:
        result = "Not Spam ✅"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run()
