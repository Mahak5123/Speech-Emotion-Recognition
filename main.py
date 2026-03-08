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
import pygame
from scipy.fftpack import fft, ifft

# ---------------- CONFIGURATION ----------------
DURATION = 3.5
SAMPLE_RATE = 22050
TARGET_VOLUME_LIVE = 0.8

MODEL_JSON = "CNN_model.json"
MODEL_WEIGHTS = "best_model.weights.h5"
SCALER_FILE = "scaler.pickle"
ENCODER_FILE = "encoder.pickle"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
pygame.mixer.init()

# ---------------- IMPROVED NOISE FILTRATION ----------------
def spectral_subtraction(audio, noise_estimation_limit=0.1):
    audio_fft = fft(audio)
    audio_mag = np.abs(audio_fft)
    audio_phase = np.angle(audio_fft)
    noise_frames = int(len(audio_mag) * noise_estimation_limit)
    noise_estimate = np.median(audio_mag[:noise_frames])
    clean_mag = audio_mag - noise_estimate
    clean_mag = np.maximum(clean_mag, 0.02 * audio_mag)
    clean_fft = clean_mag * np.exp(1j * audio_phase)
    clean_audio = np.real(ifft(clean_fft))
    clean_audio[np.abs(clean_audio) < 0.01] = 0
    return clean_audio.astype(np.float32)

# ---------------- LOAD MODEL ----------------
def load_assets():
    print("Loading model assets...")
    try:
        with open(MODEL_JSON, "r") as json_file:
            model_json = json_file.read()
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights(MODEL_WEIGHTS)
        with open(SCALER_FILE, "rb") as f:
            scaler = pickle.load(f)
        with open(ENCODER_FILE, "rb") as f:
            encoder = pickle.load(f)
        print("✔ Model loaded successfully")
        return model, scaler, encoder
    except Exception as e:
        print("❌ Error loading model:", e)
        return None, None, None

# ---------------- RECORD AUDIO ----------------
def record_audio(duration=DURATION, fs=SAMPLE_RATE):
    print(f"\n🎤 Recording for {duration} seconds... Speak clearly")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(recording)

# ---------------- PLAY AUDIO ----------------
def play_audio_file(filepath):
    if os.path.exists(filepath):
        print("🔊 Playing audio:", os.path.basename(filepath))
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

# ---------------- FILE BROWSER ----------------
def browse_file_popup():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=[("WAV files", "*.wav")])
    root.destroy()
    return file_path

def select_project_file():
    files = [f for f in os.listdir(".") if f.endswith(".wav")]
    if not files:
        print("⚠ No wav files found")
        return None
    print("\nProject files:")
    for i, f in enumerate(files):
        print(f"[{i+1}] {f}")
    choice = int(input("Select file: ")) - 1
    return files[choice]

# ---------------- FEATURE EXTRACTION ----------------
def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr=22050, frame_length=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20, n_fft=frame_length, hop_length=hop_length)
    return np.ravel(mfccs.T)

def extract_features(data):
    return np.hstack((zcr(data), rmse(data), mfcc(data)))

# ---------------- AUDIO PROCESSING ----------------
def process_file_exact(filepath):
    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=2.5, offset=0.6)
        print("✨ Applying Noise Filtration...")
        audio = spectral_subtraction(audio)
        target = int(2.5 * SAMPLE_RATE)
        if len(audio) < target:
            audio = np.pad(audio, (0, target - len(audio)))
        else:
            audio = audio[:target]
        return audio
    except Exception as e:
        print("Error reading file:", e)
        return None

def process_live_mic(audio):
    print("✨ Applying Noise Filtration...")
    audio = spectral_subtraction(audio)
    audio, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio) < 500:
        return None
    audio = librosa.util.normalize(audio)
    audio = audio * TARGET_VOLUME_LIVE
    target = int(2.5 * SAMPLE_RATE)
    if len(audio) > target:
        start = (len(audio) - target) // 2
        audio = audio[start:start + target]
    else:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio

# ---------------- PREDICTION ----------------
def predict(model, scaler, encoder, audio):
    features = extract_features(audio).flatten()
    if features.size != 2376:
        if features.size < 2376:
            features = np.pad(features, (0, 2376 - features.size))
        else:
            features = features[:2376]
    features = features.reshape(1, -1)
    scaled = scaler.transform(features)
    final_input = np.expand_dims(scaled, axis=2)
    prediction = model.predict(final_input, verbose=0)
    return prediction[0]

# ---------------- MAIN PROGRAM ----------------
def main():
    model, scaler, encoder = load_assets()
    if model is None:
        return
    emotions = encoder.categories_[0]
    while True:
        print("\n" + "=" * 50)
        print(" SPEECH EMOTION RECOGNITION + NOISE FILTRATION ")
        print("=" * 50)
        print("[1] Live Microphone (Cleaned)")
        print("[2] Project WAV File")
        print("[3] Browse Computer")
        print("[Q] Quit")
        choice = input("\nSelect option: ").lower()
        if choice == "q":
            break
        audio = None
        if choice == "1":
            raw = record_audio()
            audio = process_live_mic(raw)
        elif choice == "2":
            file = select_project_file()
            if file:
                play_audio_file(file)
                audio = process_file_exact(file)
        elif choice == "3":
            file = browse_file_popup()
            if file:
                play_audio_file(file)
                audio = process_file_exact(file)
        else:
            print("Invalid option")
            continue
        if audio is None:
            print("⚠ Could not process audio")
            continue
        probs = predict(model, scaler, encoder, audio)
        scores = [(emotions[i], probs[i]) for i in range(len(probs))]
        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[0][0]
        print("\n" + "-" * 50)
        print(f"{'EMOTION':<12} | {'CONFIDENCE':<10}")
        print("-" * 50)
        for emo, score in scores:
            bar = "█" * int(score * 20)
            prefix = ">>" if emo == top else " "
            print(f"{prefix} {emo.upper():<10} | {score*100:6.2f}% {bar}")
        print("-" * 50)
        print("🏆 FINAL PREDICTION:", top.upper())
        nav = input("\nPress ENTER to continue or Q to quit: ")
        if nav.lower() == "q":
            break

if __name__ == "__main__":
    main()