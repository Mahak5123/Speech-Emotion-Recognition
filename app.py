import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
import pickle
import json
import io
import soundfile as sf
from scipy.fftpack import fft, ifft
import speech_recognition as sr
import pandas as pd
import time
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Healthcare Emotion Analytics",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- IMPRESSIVE LIGHT THEME CSS (NO PLOTLY NEEDED) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        background-attachment: fixed;
    }
    
    /* Remove default padding */
    .main { 
        padding: 0 !important;
    }
    
    .block-container {
        padding: 2rem 3rem !important;
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Header Styling */
    h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem !important;
        font-weight: 900 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: 2px;
    }
    
    h2 {
        font-family: 'Poppins', sans-serif;
        color: #2d3748;
        font-weight: 700 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        font-family: 'Poppins', sans-serif;
        color: #667eea;
        font-weight: 600 !important;
        margin-top: 1rem !important;
    }
    
    /* Status Badge */
    .status-badge {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 2px solid #667eea;
        border-radius: 50px;
        padding: 1rem 2rem;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        color: #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Card Styling */
    .status-box, .info-card {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(10px);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .status-box:hover, .info-card:hover {
        box-shadow: 0 12px 35px rgba(102, 126, 234, 0.25);
        transform: translateY(-2px);
    }
    
    .status-box strong {
        color: #667eea;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        height: 3.5em;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        font-family: 'Poppins', sans-serif;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
    }
    
    /* Slider Styling */
    .stSlider>div>div>div>div {
        background: linear-gradient(to right, #667eea, #764ba2);
    }
    
    /* Text and Labels */
    .stMarkdown {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Success/Info/Error Messages */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 15px;
        border-left: 5px solid;
        padding: 1.5rem;
        font-family: 'Poppins', sans-serif;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 200, 100, 0.1) 100%);
        border-left-color: #00ff88 !important;
        color: #1a5d3a;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-left-color: #667eea !important;
        color: #2d3748;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 170, 0, 0.1) 0%, rgba(255, 150, 0, 0.1) 100%);
        border-left-color: #ffaa00 !important;
        color: #663d00;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.1) 0%, rgba(255, 80, 120, 0.1) 100%);
        border-left-color: #ff3366 !important;
        color: #7a0a2e;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
        font-family: 'Orbitron', sans-serif;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3), rgba(102, 126, 234, 0.3));
        margin: 2rem 0;
    }
    
    /* Results Table */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        background: white;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    /* Caption */
    .stCaption {
        text-align: center;
        color: #718096;
        font-size: 0.9rem;
        margin-top: 2rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* History Box */
    .history-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .history-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        border: 1px solid rgba(102, 126, 234, 0.2);
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .history-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.25);
    }
    
    .emotion-emoji {
        font-size: 3rem;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .emotion-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #667eea;
        text-align: center;
        text-transform: capitalize;
    }
    
    .emotion-confidence {
        font-size: 1.2rem;
        color: #00ff88;
        text-align: center;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    /* Recording Indicator */
    .recording-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 0.5rem;
        animation: pulse 1s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
        font-weight: 300;
    }
    </style>
    
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Poppins:wght@300;400;600;700;800&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- CORE LOGIC (FROM YOUR SHARED MODEL) ---
@st.cache_resource
def load_all_assets():
    try:
        with open("CNN_model.json", "r") as f:
            model = tf.keras.models.model_from_json(f.read())
        model.load_weights("best_model.weights.h5")
        with open("scaler.pickle", "rb") as f: 
            scaler = pickle.load(f)
        with open("encoder.pickle", "rb") as f: 
            encoder = pickle.load(f)
        return model, scaler, encoder
    except:
        st.error("❌ Model assets not found! Ensure CNN_model.json, best_model.weights.h5, scaler.pickle, and encoder.pickle are in the folder.")
        return None, None, None

def spectral_subtraction(audio):
    """Advanced noise filtering"""
    audio_fft = fft(audio)
    audio_mag = np.abs(audio_fft)
    audio_phase = np.angle(audio_fft)
    noise_estimate = np.mean(audio_mag[:int(len(audio_mag) * 0.1)])
    clean_mag = np.maximum(audio_mag - (1.5 * noise_estimate), 0.1 * audio_mag)
    clean_fft = clean_mag * np.exp(1j * audio_phase)
    return np.real(ifft(clean_fft)).astype(np.float32)

# --- EMOTION ICONS ---
EMOTION_ICONS = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'sad': '😢',
    'surprise': '😲'
}

def get_emotion_icon(emotion):
    return EMOTION_ICONS.get(emotion.lower(), '😐')

def get_emotion_color(emotion):
    colors = {
        'happy': '#00ff88',
        'neutral': '#667eea',
        'sad': '#4169E1',
        'angry': '#ff3366',
        'surprise': '#ffaa00',
        'fear': '#8B0000',
        'disgust': '#228B22'
    }
    return colors.get(emotion.lower(), '#667eea')

# --- INITIALIZE SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# --- LOAD MODEL ---
model, scaler, encoder = load_all_assets()

# --- MAIN DASHBOARD ---
st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h1>🏥 Healthcare Emotion Analytics</h1>
        <p class="subtitle">Real-time emotion detection from patient speech using advanced machine learning</p>
        <div class="status-badge">
            <span class="recording-indicator"></span>
            System Active & Ready
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# --- MAIN INTERFACE ---
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown("### 🎙️ Patient Recording Interface")
    
    st.markdown("**Recording Settings**")
    duration = st.slider(
        "Recording Duration (seconds)",
        min_value=2.5,
        max_value=5.0,
        value=3.5,
        step=0.5,
        help="Select how long to record patient speech"
    )
    
    st.markdown("")
    
    # Record Button
    if st.button("🔴 START LIVE MONITORING", use_container_width=True):
        with st.spinner("🎤 Recording patient audio..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            fs = 22050
            recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            
            for i in range(100):
                time.sleep(duration / 100)
                progress_bar.progress(i / 100)
            
            sd.wait()
            audio_raw = np.squeeze(recording)
            progress_bar.progress(100)
            
            status_text.markdown("✅ Audio recorded successfully")
            st.success("✅ Recording complete! Processing audio...")
            
            # --- PROCESSING ---
            status_text.markdown("🔄 Applying advanced DSP filters...")
            filtered = spectral_subtraction(audio_raw)
            
            # Transcription Path
            status_text.markdown("📝 Transcribing speech...")
            recognizer = sr.Recognizer()
            byte_io = io.BytesIO()
            sf.write(byte_io, audio_raw, fs, format='WAV')
            byte_io.seek(0)
            try:
                with sr.AudioFile(byte_io) as source:
                    audio_trans = recognizer.record(source)
                transcript = recognizer.recognize_google(audio_trans)
            except:
                transcript = "[Audio unclear - trying emotion detection anyway]"

            # Prediction Path
            status_text.markdown("🤖 Running emotion detection model...")
            trimmed, _ = librosa.effects.trim(filtered, top_db=30)
            norm = (trimmed / np.max(np.abs(trimmed))) * 0.4
            target_len = int(2.5 * fs)
            final_audio = np.pad(norm, (0, max(0, target_len - len(norm))))[:target_len]
            
            zcr = np.squeeze(librosa.feature.zero_crossing_rate(y=final_audio))
            rmse = np.squeeze(librosa.feature.rms(y=final_audio))
            mfcc = np.ravel(librosa.feature.mfcc(y=final_audio, sr=fs, n_mfcc=20).T)
            features = np.hstack((zcr, rmse, mfcc))
            
            if features.size != 2376:
                features = np.pad(features, (0, max(0, 2376 - features.size)))[:2376]
            
            scaled = scaler.transform(features.reshape(1, -1))
            probs = model.predict(np.expand_dims(scaled, axis=2), verbose=0)[0]
            emotions = encoder.categories_[0]
            
            # --- STORE RESULTS ---
            results_df = pd.DataFrame({
                'Emotion': emotions,
                'Confidence': probs * 100
            }).sort_values(by='Confidence', ascending=False)
            
            st.session_state.current_result = {
                'transcript': transcript,
                'results': results_df,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add to history
            st.session_state.history.append({
                'timestamp': st.session_state.current_result['timestamp'],
                'emotion': results_df.iloc[0]['Emotion'],
                'confidence': results_df.iloc[0]['Confidence'],
                'transcript': transcript
            })
            
            status_text.markdown("✨ Analysis complete!")
            st.balloons()

with col2:
    st.markdown("### 📊 Clinical Analysis Results")
    
    if st.session_state.current_result:
        result = st.session_state.current_result
        results_df = result['results']
        top_emotion = results_df.iloc[0]['Emotion'].upper()
        top_confidence = results_df.iloc[0]['Confidence']
        
        # Patient Utterance Box
        st.markdown(f"""
        <div class="status-box">
            <strong style="color: #667eea;">📝 Patient Utterance</strong><br>
            <div style="font-size: 1.1rem; color: #2d3748; margin-top: 0.5rem; font-style: italic;">
            "{result['transcript']}"
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Emotion Result Box
        emotion_icon = get_emotion_icon(top_emotion)
        emotion_color = get_emotion_color(top_emotion.lower())
        
        st.markdown(f"""
        <div class="status-box">
            <div style="text-align: center;">
                <div style="font-size: 3.5rem; margin: 1rem 0;">{emotion_icon}</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #667eea; text-transform: capitalize; margin: 0.5rem 0;">{top_emotion}</div>
                <div style="font-size: 2rem; font-weight: 800; color: {emotion_color}; margin: 0.5rem 0;">{top_confidence:.1f}%</div>
                <div style="font-size: 0.9rem; color: #718096; text-transform: uppercase; letter-spacing: 1px;">Confidence Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("")
        
        # Emotion Distribution Table
        st.markdown("**Emotion Distribution Analysis**")
        
        # Create colored table
        display_df = results_df.copy()
        display_df['Confidence'] = display_df['Confidence'].apply(lambda x: f"{x:.2f}%")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Timestamp
        st.caption(f"✅ Analysis completed at {result['timestamp']}")
        
    else:
        st.info("""
        👋 Welcome to the Healthcare Emotion Analytics System!
        
        **How to use:**
        1. Set your preferred recording duration
        2. Click "START LIVE MONITORING" to begin recording
        3. Speak naturally into your microphone
        4. The system will analyze your emotion and display results
        
        **The system will:**
        - 🎤 Record your speech
        - 🔊 Apply noise filtering
        - 📝 Transcribe your words
        - 🤖 Detect your emotion using AI
        - 📊 Display confidence scores
        """)

st.markdown("---")

# --- HISTORY SECTION ---
if st.session_state.history:
    st.markdown("### 📜 Analysis History")
    
    # History stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Analyses</div>
            <div class="metric-value">{len(st.session_state.history)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_confidence = np.mean([h['confidence'] for h in st.session_state.history])
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-value">{avg_confidence:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        most_common = max(set([h['emotion'] for h in st.session_state.history]), 
                         key=[h['emotion'] for h in st.session_state.history].count)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Most Common</div>
            <div class="metric-value">{get_emotion_icon(most_common)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Sessions</div>
            <div class="metric-value">{len(st.session_state.history)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # History grid
    st.markdown("**Recent Analyses**")
    
    cols = st.columns(3)
    for idx, item in enumerate(st.session_state.history[:9]):
        with cols[idx % 3]:
            emotion_icon = get_emotion_icon(item['emotion'])
            emotion_color = get_emotion_color(item['emotion'].lower())
            
            st.markdown(f"""
            <div class="history-card">
                <div class="emotion-emoji">{emotion_icon}</div>
                <div class="emotion-name">{item['emotion']}</div>
                <div class="emotion-confidence">{item['confidence']:.1f}% Confidence</div>
                <div style="font-size: 0.85rem; color: #718096; text-align: center; margin-top: 0.8rem;">
                    {item['timestamp']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("")
    
    # Clear history
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.session_state.current_result = None
        st.rerun()

st.markdown("---")

# --- FOOTER ---
st.markdown("""
    <div style="text-align: center; color: #718096; font-family: 'Poppins', sans-serif; margin-top: 3rem;">
        <strong>🏫 Final Year ECE Major Project | B.Tech 2026 </strong><br>
        <span style="font-size: 0.9rem;">
        Harnessing IoT & Machine Learning for Patient Speech Emotion Recognition<br>
        © 2026 All Rights Reserved
        </span>
    </div>
""", unsafe_allow_html=True)