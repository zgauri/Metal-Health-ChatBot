from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and tokenizer
model = load_model("mental_health_rnn.h5")
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

max_len = 100  # Same max length as used during model training

def preprocess_text(text):
    """Clean text by removing emojis and formatting."""
    text = re.sub(r'[\U0001F600-\U0001F64F]', '', text)  # Remove emojis
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text.lower().strip()  # Lowercase and strip leading/trailing spaces

def classify_message(text):
    """Classify message into Controlled, Sub-Controlled, or Suicidal."""
    text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    category = ['Controlled', 'Sub-Controlled', 'Suicidal'][np.argmax(prediction)]
    return category

@app.route('/chat', methods=['POST'])
def chat():
    """Receive a chat message and classify its category."""
    data = request.json
    user_input = data.get('message', '')
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    
    category = classify_message(user_input)
    return jsonify({"category": category})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
