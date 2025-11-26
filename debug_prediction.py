"""
Standalone debug script to test emotion prediction without recording.
Useful for diagnosing scaler, feature extraction, and model issues.
"""

import os
import json
import pickle
import numpy as np
import librosa
import tensorflow as tf
import warnings

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# Configuration
MODEL_JSON = 'CNN_model.json'
MODEL_WEIGHTS = 'best_model.weights.h5'
SCALER_FILE = 'scaler.pickle'
ENCODER_FILE = 'encoder.pickle'
SAMPLE_RATE = 22050
DURATION = 3.0

def load_assets():
    print("Loading assets...")
    with open(MODEL_JSON, 'r') as f:
        model = tf.keras.models.model_from_json(f.read())
    model.load_weights(MODEL_WEIGHTS)
    
    with open(SCALER_FILE, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(ENCODER_FILE, 'rb') as f:
        encoder = pickle.load(f)
    
    print("✔ Assets loaded.\n")
    
    # Print scaler details
    print("=== SCALER DEBUG ===")
    print(f"n_features_in_: {scaler.n_features_in_}")
    print(f"mean_ (first 10): {scaler.mean_[:10]}")
    print(f"scale_ (first 10): {scaler.scale_[:10]}")
    print(f"var_ (first 10): {scaler.var_[:10]}")
    
    # Print encoder details
    print("\n=== ENCODER DEBUG ===")
    print(f"categories_: {encoder.categories_}")
    
    return model, scaler, encoder

def extract_features(audio_data, sample_rate, n_mfcc=40):
    """Extract MFCC features and pad to 2376."""
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
    result = mfccs.flatten()
    
    target_length = 2376
    if len(result) < target_length:
        result = np.pad(result, (0, target_length - len(result)), mode='constant')
    else:
        result = result[:target_length]
    
    return result

def test_prediction(model, scaler, encoder, audio_data):
    """Test prediction on given audio (no preprocessing)."""
    print("\n=== FEATURE EXTRACTION ===")
    features = extract_features(audio_data, SAMPLE_RATE)
    print(f"Raw features shape: {features.shape}")
    print(f"Raw features (first 10): {features[:10]}")
    print(f"Raw features min/max: {features.min():.4f} / {features.max():.4f}")
    
    # Reshape and scale
    features = features.reshape(1, -1)
    scaled_features = scaler.transform(features)
    
    print("\n=== SCALING ===")
    print(f"Scaled features (first 10): {scaled_features[0][:10]}")
    print(f"Scaled features min/max: {scaled_features.min():.4f} / {scaled_features.max():.4f}")
    
    # Model input
    final_input = np.expand_dims(scaled_features, axis=2)
    print(f"\nModel input shape: {final_input.shape}")
    
    # Predict
    prediction = model.predict(final_input, verbose=0)
    
    print("\n=== PREDICTION ===")
    print(f"Raw probabilities: {prediction[0]}")
    
    # Decode
    categories = encoder.categories_[0]
    predicted_idx = np.argmax(prediction[0])
    predicted_label = categories[predicted_idx]
    
    print(f"\nCategories: {categories}")
    print(f"Predicted index: {predicted_idx}")
    print(f"Predicted emotion: {predicted_label}")
    print(f"Confidence: {prediction[0][predicted_idx]:.4f}")

def main():
    model, scaler, encoder = load_assets()
    
    # Test 1: Silent audio (all zeros)
    print("\n" + "="*60)
    print("TEST 1: Silent audio (all zeros)")
    print("="*60)
    silent = np.zeros(int(SAMPLE_RATE * DURATION), dtype='float32')
    test_prediction(model, scaler, encoder, silent)
    
    # Test 2: White noise (random audio)
    print("\n" + "="*60)
    print("TEST 2: White noise (random audio)")
    print("="*60)
    noise = np.random.randn(int(SAMPLE_RATE * DURATION)).astype('float32') * 0.1
    test_prediction(model, scaler, encoder, noise)
    
    # Test 3: Sine wave (simple frequency)
    print("\n" + "="*60)
    print("TEST 3: Sine wave at 440 Hz")
    print("="*60)
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), dtype='float32')
    sine = np.sin(2 * np.pi * 440 * t).astype('float32')
    test_prediction(model, scaler, encoder, sine)

if __name__ == "__main__":
    main()
