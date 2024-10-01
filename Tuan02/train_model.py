# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
import joblib

# Load the dataset
df = pd.read_csv("Data/Education.csv")
text, label = df['Text'], df['Label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    text, label, test_size=0.2, random_state=42
)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Convert sparse matrices to dense
X_train_vect = X_train_vect.toarray()
X_test_vect = X_test_vect.toarray()

# Initialize the classifiers
bernoulli = BernoulliNB()
multinomial = MultinomialNB()

# Train the classifiers
bernoulli.fit(X_train_vect, y_train)
multinomial.fit(X_train_vect, y_train)

# Save the models and vectorizer to disk
joblib.dump(bernoulli, 'bernoulli_nb_model.joblib')
joblib.dump(multinomial, 'multinomial_nb_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Models and vectorizer have been saved successfully.")
