texts = [
    "I feel completely at ease and relaxed today. Everything is going smoothly.",
    "The weekend was peaceful and I feel refreshed and calm.",
    "Just enjoying a quiet moment. Nothing is troubling me.",
    "A wonderful, stress-free day. I am feeling great.",
    "I am totally calm and happy with the current situation.",

    "It's just a normal workday, checking emails and running routine tasks.",
    "I feel a bit tired, but the day is fine.",
    "The weather is cloudy, but I have no strong feelings about my tasks.",
    "I have a few things to do, but it's manageable.",
    "Just another day. Nothing exciting or worrying.",

    "I am so incredibly overwhelmed and stressed out with all this work.",
    "The deadline is tomorrow and I seriously can't handle this mounting pressure.",
    "Feeling a lot of pressure and significant anxiety about the presentation.",
    "I'm completely overwhelmed and frustrated with the constant demands.",
    "This project is too much for me to bear; I need a massive break.",
    "I'm at my limit. Everything is falling apart and I'm panicking.",
    "I have a huge headache and feel like crying from the stress."
]

labels = [
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2
]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

print("--- Model Training Initiated ---")

model = make_pipeline(TfidfVectorizer(), MultinomialNB())

model.fit(texts, labels)

joblib.dump(model, "stress_model.pkl")

print("Model training completed & file saved!")
