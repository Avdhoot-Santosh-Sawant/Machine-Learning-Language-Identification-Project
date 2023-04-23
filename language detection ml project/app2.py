import re
import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
language = pd.read_csv('Language Detection.csv')

# Preprocess the text data
X = language["Text"]
y = language["Language"]
le = LabelEncoder()
y = le.fit_transform(y)
text_list = []
for text in X:
        text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
        text = re.sub(r'[[]]', ' ', text)
        text = text.lower()
        text_list.append(text)

# Vectorize the text data
cv = CountVectorizer()
X = cv.fit_transform(text_list).toarray()

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Train the model
model = MultinomialNB()
model.fit(x_train, y_train)

# Predict on the test set and calculate accuracy
y_prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction)
confusion_m = confusion_matrix(y_test, y_prediction)
print("The accuracy is :",accuracy)

# Save the model and vectorizer to disk
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(cv, open("vectorizer.pkl", "wb"))

# Load the model and vectorizer in the Flask app
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('vectorizer.pkl', 'rb'))

# Define the predict function
@app.route('/predict',methods=['POST'])
def predict():
    # Get the text from the POST request
    text = request.json['text']
    
    # Preprocess the text
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    
    # Vectorize the text
    text_vectorized = cv.transform([text]).toarray()
    
    # Make the prediction
    prediction = model.predict(text_vectorized)
    
    # Reverse the label encoding
    predicted_language = le.inverse_transform(prediction)[0]
    
    # Return the predicted language
    return jsonify({'language': predicted_language})

# Define the home page
@app.route('/')
def home():
    return render_template('index.html')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)