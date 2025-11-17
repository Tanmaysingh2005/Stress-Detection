🧠 Stress Detection (ML + Flask)

A lightweight machine learning–based Stress Detection System that analyzes user text and predicts whether the user is Stressed or Not Stressed.

🚀 Features

Simple & clean web interface

Flask-based backend

ML model using TF-IDF + Logistic Regression

Real-time predictions

📦 Project Structure
├── index.html
├── app.py
├── train.py
├── model.pkl
├── vectorizer.pkl
└── requirements.txt

▶️ Run Locally
pip install -r requirements.txt
python app.py


Visit: http://127.0.0.1:5000

🎯 How It Works

User enters text

Text → TF-IDF Vectorizer

Model predicts stress level

Result displayed on UI

🛠 Technologies

Python, Flask

Scikit-learn (TF-IDF, Logistic Regression)

HTML, CSS, JS

📘 Training (Optional)
python train.py
