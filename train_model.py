import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

# Load the preprocessed dataset
dataset_path = "preprocessed_dataset.csv"
data = pd.read_csv(dataset_path)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['clean_text'])

# Convert text to sequences
sequences = tokenizer.texts_to_sequences(data['clean_text'])

# Pad sequences
max_len = 100  # Define max input length
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# Encode labels
y = data['label'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save tokenizer for later use
with open("tokenizer.pkl", "wb") as file:
    pickle.dump(tokenizer, file)

# Define the RNN model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')  # Output layer for 3 categories
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# Save the trained model
model.save("mental_health_rnn.h5")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 
