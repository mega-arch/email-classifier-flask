# flask_app.py
from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from werkzeug.security import generate_password_hash, check_password_hash
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Dummy user credentials (replace with DB in production)
users = {
    "admin": generate_password_hash("admin123")
}

# Load model and label encoder
model: Pipeline = joblib.load('email_classifier.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session['username'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and check_password_hash(users[username], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    subject = request.form['subject']
    body = request.form['body']
    text = subject + " " + body
    prediction = model.predict([text])[0]
    category = label_encoder.inverse_transform([prediction])[0]

    proba = model.predict_proba([text])[0]
    probas = dict(zip(label_encoder.inverse_transform(range(len(proba))), [round(p * 100, 2) for p in proba]))

    return render_template('index.html', prediction=category, probabilities=probas, username=session['username'])

if __name__ == '__main__':
    app.run(debug=True)
