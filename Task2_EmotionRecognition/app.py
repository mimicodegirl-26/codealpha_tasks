import streamlit as st
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# ========== CONFIG (must match training) ==========
SAMPLE_RATE = 22050
N_MFCC = 60
MAX_PAD_LEN = 200

# ========== LOAD MODEL & ENCODER ==========
model = load_model('emotion_model.h5')
le = joblib.load('label_encoder.pkl')

def extract_features(audio, sr):
    """Extract MFCC + delta + delta2 and pad/truncate to MAX_PAD_LEN."""
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    # Delta and delta-delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # Stack vertically -> (180, time_frames)
    features = np.vstack([mfcc, delta, delta2])

    # Pad or truncate to fixed length (time axis)
    if features.shape[1] < MAX_PAD_LEN:
        pad_width = MAX_PAD_LEN - features.shape[1]
        features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :MAX_PAD_LEN]

    # Transpose to (time_steps, features) for the model: (200, 180)
    features = features.T
    return features

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="Emotion Recognition", layout="centered")
st.title("🎤 Emotion Recognition from Speech")
st.write("Upload a `.wav` file and the model will predict the emotion.")

uploaded_file = st.file_uploader("Choose a WAV file", type=['wav'])
if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    # Load and preprocess
    audio, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
    features = extract_features(audio, sr)
    # Add batch dimension
    features = np.expand_dims(features, axis=0)   # shape (1, 200, 180)

    # Predict
    pred_probs = model.predict(features)
    pred_idx = np.argmax(pred_probs)
    emotion = le.inverse_transform([pred_idx])[0]
    confidence = np.max(pred_probs)

    st.success(f"**Predicted Emotion:** {emotion}  (confidence: {confidence:.2%})")