import pickle
import re
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import numpy as np
app = Flask(__name__)

# Load the model and vectorizer from disk
model = pickle.load(open("model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

le = LabelEncoder()
le.classes_ = np.load('classes.npy', allow_pickle=True)

# Define a function to preprocess text


def preprocess_text(text):
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    return text

# Define a function to predict the language of text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    text = preprocess_text(text)
    text_vector = cv.transform([text]).toarray()
    prediction = model.predict(text_vector)[0]
    predicted_language = le.inverse_transform([prediction])[0]
    return render_template('index.html', predicted_language=predicted_language, text=text)


if __name__ == '__main__':
    app.run(debug=True)
