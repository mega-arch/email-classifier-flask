# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# Sample data
data = {
    "text": [
        "Congratulations! You’ve won a free ticket.",
        "Let's schedule our team meeting for tomorrow.",
        "Your invoice for the recent purchase is attached.",
        "Limited offer! Buy one get one free.",
        "Project deadline is next Friday. Please review.",
    ],
    "label": ["spam", "work", "finance", "spam", "work"]
}
df = pd.DataFrame(data)

# Encode labels
label_encoder = LabelEncoder()
df['encoded'] = label_encoder.fit_transform(df['label'])

# Create pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Train
model.fit(df['text'], df['encoded'])

# Save model and encoder
joblib.dump(model, 'email_classifier.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
