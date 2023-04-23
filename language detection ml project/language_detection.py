import re
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Load the dataset
language = pd.read_csv('Language Detection.csv')

# Preprocess the text data
X = language["Text"]
y = language["Language"]
le = LabelEncoder()
y = le.fit_transform(y)
np.save('classes.npy', le.classes_)

text_list = []
for text in X:
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    text_list.append(text)

# Vectorize the text data
cv = CountVectorizer()
X = cv.fit_transform(text_list).toarray()

# Save the vectorizer to disk
pickle.dump(cv, open("vectorizer.pkl", "wb"))

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Train the model
model = MultinomialNB()
model.fit(x_train, y_train)

# Save the model to disk
pickle.dump(model, open("model.pkl", "wb"))
