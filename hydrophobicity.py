import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load your dataset
csv_file_path = "HumanPlasma2023-04.csv"
df = pd.read_csv(csv_file_path)

# Extract amino acid sequences and SSRCalc_relative_hydrophobicity
amino_acid_sequences = df['peptide_sequence'].tolist()
hydrophobicity_values = df['SSRCalc_relative_hydrophobicity'].values

# Tokenize sequences into one-hot encoded vectors
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(amino_acid_sequences)

# Convert amino acid sequences to numerical sequences
numerical_sequences = tokenizer.texts_to_sequences(amino_acid_sequences)

# Pad sequences with zeros
max_seq_length = 50
padded_sequences = pad_sequences(numerical_sequences, padding='post', maxlen=max_seq_length)

# One-hot encode the padded sequences
one_hot_encoded_sequences = tf.keras.utils.to_categorical(padded_sequences, num_classes=len(tokenizer.word_index) + 1)

# Split the data into training, validation, and test sets
split_ratio_train = 0.8
split_ratio_val = 0.1
split_index_train = int(len(one_hot_encoded_sequences) * split_ratio_train)
split_index_val = int(len(one_hot_encoded_sequences) * (split_ratio_train + split_ratio_val))

X_train = one_hot_encoded_sequences[:split_index_train]
y_train = hydrophobicity_values[:split_index_train]

X_val = one_hot_encoded_sequences[split_index_train:split_index_val]
y_val = hydrophobicity_values[split_index_train:split_index_val]

X_test = one_hot_encoded_sequences[split_index_val:]
y_test = hydrophobicity_values[split_index_val:]

print("Size of Training Set:", X_train.shape[0])
print("Size of Validation Set:", X_val.shape[0])
print("Size of Test Set:", X_test.shape[0])

# Build the regression model with one-hot encoded input
def build_regression_model(max_seq_length, vocab_size):

    inputs = tf.keras.Input(shape=(max_seq_length, vocab_size), batch_size=32, dtype=tf.float32)

    #x = layers.Embedding(input_dim=vocab_size + 1,
    #                     output_dim=4,
    #                    input_length=max_seq_length)(inputs)

    # Flatten the one-hot encoded sequences
    x = layers.Flatten()(inputs)
    #x = layers.Flatten()(x)

    # Dense layers with dropout and regularization
    x = layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Dense(units=128, activation='relu')(x)

    # Output layer for regression
    outputs = layers.Dense(units=1, activation="linear")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hydrophobicity_predictor")
    return model

# Model parameters
vocab_size = len(tokenizer.word_index) + 1  # Add 1 for padding

# Build the regression model
regression_model = build_regression_model(max_seq_length, vocab_size)

# Compile the model for regression
regression_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2*1e-5), loss="mean_squared_error", metrics=["mae"],)

# Print the model summary
print("Model summary")
regression_model.summary()

# Callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the regression model
history = regression_model.fit(X_train, y_train, epochs=30, batch_size=32, 
                                validation_data=(X_val, y_val), 
                                callbacks=[early_stopping])

# Evaluate on the test set
print("\nEvaluating on test set")
regression_model.evaluate(X_test, y_test)



# Plot training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

