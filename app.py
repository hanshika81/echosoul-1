# app.py
import os
import json
import base64
import hashlib
import subprocess
import sys
from io import BytesIO
from datetime import datetime
from typing import Tuple, List

import streamlit as st
import openai
import pandas as pd
from textblob import TextBlob
from cryptography.fernet import Fernet
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from dotenv import load_dotenv
# Ensure NLTK and TextBlob corpora are available
import nltk
from textblob import download_corpora

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    download_corpora.download_all()
except Exception:
    pass
# -------------------------------------------
# Auto-install TextBlob/NLTK data if missing
# -------------------------------------------
import nltk
from textblob import download_corpora

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("wordnet")
    nltk.download("omw-1.4")
    nltk.download("brown")
    download_corpora.download_all()

# -------------------------------------------
# Try to import spaCy (optional)
# -------------------------------------------
try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

# -------------------------------------------
# Load environment variables
# -------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AGORA_APP_ID = os.getenv("AGORA_APP_ID", "")
AGORA_APP_CERTIFICATE = os.getenv("AGORA_APP_CERTIFICATE", "")

openai.api_key = OPENAI_API_KEY or None

st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("🌙 EchoSoul — your adaptive AI companion")

# -------------------------------------------
# Emotion analysis helper
# -------------------------------------------
def analyze_emotion(text: str) -> str:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.2:
        return "happy 😊"
    elif polarity < -0.2:
        return "sad 😔"
    else:
        return "neutral 😐"

# -------------------------------------------
# Chat memory storage
# -------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------------------
# Chat function
# -------------------------------------------
def generate_response(user_input: str) -> str:
    try:
        emotion = analyze_emotion(user_input)
        prompt = f"The user seems {emotion}. Respond like an empathetic, adaptive companion."
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# -------------------------------------------
# WebRTC Call (Voice Chat)
# -------------------------------------------
def audio_streamer():
    class AudioProcessor(AudioProcessorBase):
        def recv_audio_frame(self, frame):
            return frame
    webrtc_streamer(
        key="echosoul-call",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
    )

# -------------------------------------------
# Sidebar Navigation
# -------------------------------------------
st.sidebar.title("EchoSoul")
st.sidebar.write("Adaptive companion — chat, call, remember.")
mode = st.sidebar.radio("Mode", ["Chat", "Chat history", "Life timeline", "Vault", "Call", "About"])
show_env = st.sidebar.checkbox("Show env debug (hide keys)")

if show_env:
    st.sidebar.json({
        "OPENAI_API_KEY": bool(OPENAI_API_KEY),
        "AGORA_APP_ID": bool(AGORA_APP_ID),
        "AGORA_APP_CERTIFICATE": bool(AGORA_APP_CERTIFICATE)
    })

# -------------------------------------------
# Page Logic
# -------------------------------------------
if mode == "Chat":
    st.subheader("💬 Chat with EchoSoul")
    user_text = st.text_area("Your message", placeholder="Share your thoughts...")
    if st.button("Send"):
        if user_text.strip():
            response = generate_response(user_text)
            st.session_state.chat_history.append((user_text, response))
            st.markdown(f"**You:** {user_text}")
            st.markdown(f"**EchoSoul:** {response}")
        else:
            st.warning("Please enter a message.")
elif mode == "Chat history":
    st.subheader("🧠 Chat History")
    if st.session_state.chat_history:
        for user, bot in st.session_state.chat_history:
            st.markdown(f"👤 **You:** {user}")
            st.markdown(f"🤖 **EchoSoul:** {bot}")
    else:
        st.info("No chat history yet.")
elif mode == "Life timeline":
    st.subheader("📜 Life Timeline")
    st.write("EchoSoul will build your memory timeline here soon...")
elif mode == "Vault":
    st.subheader("🔐 Memory Vault")
    st.write("Personal memories and adaptive data storage coming soon.")
elif mode == "Call":
    st.subheader("📞 Live Voice Chat")
    audio_streamer()
elif mode == "About":
    st.subheader("ℹ️ About EchoSoul")
    st.markdown("EchoSoul is your AI companion — adaptive, emotional, and ever-learning 💫")
