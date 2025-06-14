# 📧 Email Classifier with Flask
A web application to classify emails into categories like Spam, Work, Promotions using Machine Learning.

## 🚀 Features
- Secure login system (username: admin, password: admin123)
- Email classification using Naive Bayes
- Confidence scores for predictions

## 📦 Installation
```bash
git clone https://github.com/yourusername/email-classifier-flask.git
cd email-classifier-flask
pip install -r requirements.txt
python flask_app.py
```

## 🌐 Deploy on Render
1. Push to GitHub
2. Go to [https://render.com](https://render.com)
3. New Web Service → Connect GitHub → Select repo
4. Add build command: `pip install -r requirements.txt`
5. Start command: `python flask_app.py`
6. Add environment: `PYTHON_VERSION=3.10`
