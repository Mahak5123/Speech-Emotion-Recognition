import os
import json
import pickle
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import warnings
import tkinter as tk
from tkinter import filedialog
import pygame  # NEW: For playing back audio files

# --- CONFIGURATION ---
DURATION = 3.5
SAMPLE_RATE = 22050
TARGET_VOLUME_LIVE = 0.6

# Files
MODEL_JSON = 'CNN_model.json'
MODEL_WEIGHTS = 'best_model.weights.h5'
SCALER_FILE = 'scaler.pickle'
ENCODER_FILE = 'encoder.pickle'

# Suppress Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")

# Initialize Pygame Mixer for Audio Playback
pygame.mixer.init()

def load_assets():
    print("Loading model assets...")
    try:
        with open(MODEL_JSON, 'r') as json_file:
            loaded_model_json = json_file.read()
        model = tf.keras.models.model_from_json(loaded_model_json)
        model.load_weights(MODEL_WEIGHTS)
        
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        with open(ENCODER_FILE, 'rb') as f:
            encoder = pickle.load(f)
            
        print("✔ Model loaded successfully.")
        return model, scaler, encoder
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        return None, None, None

# --- INPUT METHODS ---
def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"🎤 Recording for {duration}s... (Speak Now)")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(recording)

def browse_file_popup():
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    print("📂 Opening File Explorer...")
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
    )
    root.destroy()
    return file_path

def select_project_file():
    files = [f for f in os.listdir('.') if f.lower().endswith('.wav')]
    if not files:
        print("⚠ No .wav files found.")
        return None
    print("\n📂 Project Files:")
    for i, f in enumerate(files):
        print(f"  [{i+1}] {f}")
    selection = input("\n> Enter file number: ")
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(files): return files[idx]
    except: pass
    print("❌ Invalid selection.")
    return None

# --- AUDIO PLAYBACK ---
def play_audio_file(filepath):
    """Plays the audio file so the panel can hear it."""
    if os.path.exists(filepath):
        print(f"🔊 Playing audio: {os.path.basename(filepath)}...")
        try:
            pygame.mixer.music.load(filepath)
            pygame.mixer.music.play()
            # Wait for playback to finish (optional, but good for demo flow)
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"⚠ Could not play audio: {e}")
    else:
        print("⚠ File not found for playback.")

# --- FEATURE EXTRACTION ---
def zcr(data, frame_length, hop_length):
    zcr_val = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_val)

def rmse(data, frame_length=2048, hop_length=512):
    rmse_val = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_val)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_result = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, n_fft=frame_length, hop_length=hop_length)
    return np.squeeze(mfcc_result.T) if not flatten else np.ravel(mfcc_result.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((
        result,
        zcr(data, frame_length, hop_length),
        rmse(data, frame_length, hop_length),
        mfcc(data, sr, frame_length, hop_length)
    ))
    return result

# --- PROCESSING ---
def process_file_exact(filepath):
    try:
        # Load exactly as trained
        d, s_rate = librosa.load(filepath, sr=SAMPLE_RATE, duration=2.5, offset=0.6)
        
        target_samples = int(2.5 * SAMPLE_RATE)
        if len(d) < target_samples:
            padding = target_samples - len(d)
            d = np.pad(d, (0, padding), mode='constant')
        elif len(d) > target_samples:
            d = d[:target_samples]
            
        return d
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def process_live_mic(audio_data):
    # Trim Silence
    audio_trimmed, _ = librosa.effects.trim(audio_data, top_db=25)
    if len(audio_trimmed) < 100: return None

    # Normalize & Volume Control
    audio_normalized = librosa.util.normalize(audio_trimmed)
    audio_balanced = audio_normalized * TARGET_VOLUME_LIVE

    # Fit to 2.5s
    target_samples = int(2.5 * SAMPLE_RATE)
    if len(audio_balanced) > target_samples:
        start = (len(audio_balanced) - target_samples) // 2
        audio_final = audio_balanced[start : start + target_samples]
    else:
        padding = target_samples - len(audio_balanced)
        audio_final = np.pad(audio_balanced, (0, padding), mode='constant')
        
    return audio_final

def predict(model, scaler, encoder, audio):
    features = extract_features(audio)
    
    # Shape Enforcement
    expected_shape = (1, 2376)
    features = features.flatten()
    
    if features.size != 2376:
        if features.size < 2376:
            features = np.pad(features, (0, 2376 - features.size), mode='constant')
        else:
            features = features[:2376]
            
    features = features.reshape(1, -1)

    try:
        scaled_features = scaler.transform(features)
        final_input = np.expand_dims(scaled_features, axis=2)
        prediction = model.predict(final_input, verbose=0)
        return prediction[0]
    except Exception as e:
        print(f"Err in prediction: {e}")
        return None

# --- MAIN ---
def main():
    model, scaler, encoder = load_assets()
    if not model: return
    categories = encoder.categories_[0]

    while True:
        print("\n" + "="*50)
        print(" SPEECH EMOTION RECOGNITION SYSTEM")
        print("="*50)
        print(" [1] Live Microphone")
        print(" [2] Select from Project Folder")
        print(" [3] Browse Computer")
        print(" [Q] Quit")
        
        choice = input("\n> Select Option: ").strip().lower()
        if choice == 'q': break
        
        audio = None
        
        # --- PATH SELECTION ---
        if choice == '1': # LIVE
            raw_rec = record_audio()
            audio = process_live_mic(raw_rec)
            
        elif choice == '2': # PROJECT LIST
            filename = select_project_file()
            if filename:
                # PLAY AUDIO FOR PANEL
                play_audio_file(filename) 
                print(f"▶ Processing: {filename}")
                audio = process_file_exact(filename)
                
        elif choice == '3': # POPUP
            filename = browse_file_popup()
            if filename:
                # PLAY AUDIO FOR PANEL
                play_audio_file(filename)
                print(f"▶ Processing: {filename}")
                audio = process_file_exact(filename)
        else:
            print("❌ Invalid option.")
            continue

        # --- PREDICT ---
        if audio is not None:
            probs = predict(model, scaler, encoder, audio)
            
            if probs is not None:
                scores = [(categories[i], probs[i]) for i in range(len(probs))]
                scores.sort(key=lambda x: x[1], reverse=True)
                top_emo = scores[0][0]
                
                print("\n" + "-" * 50)
                print(f"{'EMOTION':<12} | {'CONFIDENCE':<10} | {'VISUALIZATION'}")
                print("-" * 50)
                for emo, score in scores:
                    bar_len = int(score * 20)
                    bar = "█" * bar_len + "░" * (20 - bar_len)
                    prefix = ">>" if emo == top_emo else "  "
                    print(f"{prefix} {emo.upper():<10} | {score*100:6.2f}%    | {bar}")
                print("-" * 50)
                print(f"🏆 FINAL PREDICTION: {top_emo.upper()}")
                
                # --- NEW EXIT LOGIC ---
                nav = input("\nPress [Enter] to continue or 'q' to quit: ")
                if nav.lower() == 'q':
                    print("Exiting...")
                    break
            else:
                print("⚠ Error: Audio processing failed.")

if __name__ == "__main__":
    main()