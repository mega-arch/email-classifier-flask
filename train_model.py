# train_model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Sample data — replace with your dataset
data = {
    'text': ["Win money now!", "Hello friend", "Get free offer", "Meeting at 10", "Claim prize now"],
    'label': [1, 0, 1, 0, 1]  # 1=Spam, 0=Not Spam
}

df = pd.DataFrame(data)

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved.")
