from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/', methods=['GET', 'POST'])
def classify():
    prediction = None
    if request.method == 'POST':
        subject = request.form.get('subject', '')
        body = request.form.get('body', '')
        full_text = subject + " " + body

        # Vectorize and predict
        input_vector = vectorizer.transform([full_text])
        result = model.predict(input_vector)[0]
        prediction = 'Spam' if result == 1 else 'Not Spam'

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
