from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model("mental_health_rnn.h5")
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Parameters
max_len = 100  # Same max length as used during training

# Preprocessing and classification function
def classify_message(text):
    """Classify input text into mental health categories."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    categories = ['Controlled', 'Sub-Controlled', 'Suicidal']
    return categories[np.argmax(prediction)]

# API Endpoint for classification
@app.route('/chat', methods=['POST'])
def chat():
    """API endpoint to classify user messages."""
    data = request.json
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    category = classify_message(user_input)
    return jsonify({"category": category})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
