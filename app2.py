import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

# --- Configuration ---
MODEL_FILE = 'stress_model.joblib'
VECTORIZER_FILE = 'vectorizer.joblib'
LABEL_MAP = {0: 'Calm', 1: 'Neutral', 2: 'Stressed'}

# Initialize Flask app
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Global variables for the model and vectorizer
model = None
vectorizer = None

# --- Model Training/Loading Utilities ---

def train_model():
    """
    Trains and saves a simple Logistic Regression model and TfidfVectorizer.
    Expanded synthetic dataset for slightly better demo coverage.
    """
    print("--- Model Training Initiated ---")
    
    # 1. Create a more diverse synthetic dataset
    texts = [
        # Calm (Label 0)
        "I feel completely at ease and relaxed today. Everything is going smoothly.",
        "The weekend was peaceful and I feel refreshed and calm.",
        "Just enjoying a quiet moment. Nothing is troubling me.",
        "A wonderful, stress-free day. I am feeling great.",
        "I am totally calm and happy with the current situation.",

        # Neutral (Label 1)
        "It's just a normal workday, checking emails and running routine tasks.",
        "I feel a bit tired, but the day is fine.",
        "The weather is cloudy, but I have no strong feelings about my tasks.",
        "I have a few things to do, but it's manageable.",
        "Just another day. Nothing exciting or worrying.",

        # Stressed (Label 2)
        "I am so incredibly overwhelmed and stressed out with all this work.",
        "The deadline is tomorrow and I seriously can't handle this mounting pressure.",
        "Feeling a lot of pressure and significant anxiety about the presentation.",
        "I'm completely overwhelmed and frustrated with the constant demands.",
        "This project is too much for me to bear; I need a massive break.",
        "I'm at my limit. Everything is falling apart and I'm panicking.",
        "I have a huge headache and feel like crying from the stress."
    ]
    
    # Corresponding labels: 0=Calm, 1=Neutral, 2=Stressed
    labels = [0, 0, 0, 0, 0, 
              1, 1, 1, 1, 1, 
              2, 2, 2, 2, 2, 2, 2]

    # 2. Vectorize the text
    try:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
    except Exception as e:
        print(f"Error during vectorization: {e}")
        return None, None

    # 3. Train the classifier
    try:
        model = LogisticRegression(max_iter=1000) # Increased max_iter for stability
        model.fit(X, labels)
    except Exception as e:
        print(f"Error during model training: {e}")
        return None, None

    # 4. Save the model and vectorizer
    try:
        joblib.dump(model, MODEL_FILE)
        joblib.dump(vectorizer, VECTORIZER_FILE)
        print(f"Model and Vectorizer saved successfully.")
    except Exception as e:
        print(f"Error saving model/vectorizer: {e}")
        return None, None
        
    print("--- Model Training Complete ---")
    return model, vectorizer

def load_model():
    """
    Loads the model and vectorizer from disk. If files don't exist, trains a new one.
    """
    global model, vectorizer
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        print(f"Model file '{MODEL_FILE}' or vectorizer file '{VECTORIZER_FILE}' not found.")
        model, vectorizer = train_model()
    else:
        try:
            print("Loading existing model and vectorizer...")
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECTORIZER_FILE)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Retraining...")
            model, vectorizer = train_model()
            
    return model, vectorizer

# Load the model on server startup
load_model()

# --- API Endpoints ---

@app.route('/status', methods=['GET'])
def get_status():
    """
    Returns the status of the server and model loading.
    """
    status = "OK" if model and vectorizer else "ERROR"
    return jsonify({
        'status': status,
        'model_loaded': bool(model),
        'vectorizer_loaded': bool(vectorizer),
        'message': 'Stress Detection API operational.' if status == 'OK' else 'Model files missing or failed to load/train.'
    }), 200 if status == 'OK' else 503


@app.route('/predict', methods=['POST'])
def predict_stress():
    """
    Receives text and returns a prediction (Calm, Neutral, Stressed) and confidence.
    """
    global model, vectorizer
    if not model or not vectorizer:
        return jsonify({'error': 'Model is not loaded or trained yet. Server status 500.'}), 500

    try:
        data = request.json
        text = data.get('text')
        
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
            return jsonify({'error': 'Invalid or insufficient text provided for analysis.'}), 400
        
        # 1. Preprocess the input text
        text_vec = vectorizer.transform([text])
        
        # 2. Make prediction
        # prediction[0] is the predicted class (0, 1, or 2)
        prediction = model.predict(text_vec)
        # proba[0] is an array of probabilities for each class [P(0), P(1), P(2)]
        proba = model.predict_proba(text_vec)
        
        # 3. Format the response
        result_label = LABEL_MAP.get(prediction[0], 'Unknown')
        confidence = proba.max() * 100
        
        response = {
            'prediction': result_label,
            'confidence': f"{confidence:.2f}"
        }
        
        print(f"Prediction for '{text[:40]}...': {result_label} with {confidence:.2f}% confidence.")
        return jsonify(response)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An internal error occurred during prediction.'}), 500

# --- Run the Server ---

if __name__ == '__main__':
    # Runs the server on http://127.0.0.1:5000
    print("\n" + "="*50)
    print("      Starting Stress Detection Flask Server")
    print("      Access: http://127.0.0.1:5000")
    print("      Status Check: http://127.0.0.1:5000/status")
    print("="*50 + "\n")
    app.run(debug=True, port=5000, use_reloader=False)