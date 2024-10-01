from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained models and vectorizer at startup
bernoulli_model = joblib.load('bernoulli_nb_model.joblib')
multinomial_model = joblib.load('multinomial_nb_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_input = request.form['text']
        if user_input.strip() == "":
            prediction = "KO TOT"
        else:
            # Vectorize the user input
            user_vect = vectorizer.transform([user_input]).toarray()
            # Predict using BernoulliNB
            ans = bernoulli_model.predict(user_vect)[0]
            # Map prediction to desired output
            if ans.lower() == "positive":
                prediction = "TOT"
            elif ans.lower() == "negative":
                prediction = "KO TOT"
            else:
                prediction = "Không xác định"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

