import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the model and tokenizer
model = load_model("mental_health_rnn.h5")
with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Parameters
max_len = 100  # Same max length used during training

# Function to preprocess and classify text
def classify_message(text):
    """Classify input text into categories."""
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    categories = ['Controlled', 'Sub-Controlled', 'Suicidal']
    return categories[np.argmax(prediction)]

# Test the model
test_messages = [
    "I feel great today!",
    "I'm tired and can't go on anymore 😢",
    "I'm struggling but trying to keep it together."
]

for message in test_messages:
    category = classify_message(message)
    print(f"Message: {message}")
    print(f"Category: {category}")
    print()
