import streamlit as st
import openai
import json
import os
import spacy
import numpy as np
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(page_title="EchoSoul", layout="wide")

# Initialize OpenAI API Key
import os

openai.api_key = os.getenv("OPENAI_API_KEY")
AGORA_APP_ID = os.getenv("AGORA_APP_ID")
AGORA_APP_CERTIFICATE = os.getenv("AGORA_APP_CERTIFICATE")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Persistent memory file
MEMORY_FILE = "data/memory.json"
os.makedirs("data", exist_ok=True)
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"timeline": [], "vault": {}, "personality": "neutral"}, f)

def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)

# NLP Sentiment Analysis
def get_emotion(text):
    doc = nlp(text)
    pos_words = ["happy", "excited", "great", "love", "awesome", "joy"]
    neg_words = ["sad", "angry", "upset", "bad", "hate", "terrible"]
    score = sum(word.text.lower() in pos_words for word in doc) - sum(word.text.lower() in neg_words for word in doc)
    return ("positive" if score > 0 else "negative" if score < 0 else "neutral"), float(score)

# Assistant personality adaptation
def get_adaptive_response(user_input, emotion_label):
    memory = load_memory()
    tone = memory.get("personality", "neutral")

    base_prompt = f"You are EchoSoul, a personal AI companion that adapts to the user's emotional tone ({emotion_label}) and maintains a {tone} personality. Respond conversationally and warmly."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": base_prompt},
                {"role": "user", "content": user_input}
            ],
        )
        message = response.choices[0].message["content"]
    except Exception:
        message = "I’m having trouble reaching my brain right now 😔. Try again in a moment."
    return message

# UI Layout
st.sidebar.title("EchoSoul")
mode = st.sidebar.radio("Mode", ["Chat", "Chat history", "Life timeline", "Vault", "Call", "About"])
st.sidebar.text_input("OpenAI API Key", type="password", key="api_input")

# Mode: Chat
if mode == "Chat":
    st.title("💬 Chat with EchoSoul")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_area("Message", key="input_area")
    if st.button("Send"):
        if user_input.strip():
            emotion, score = get_emotion(user_input)
            response = get_adaptive_response(user_input, emotion)
            st.session_state.chat_history.append({"user": user_input, "assistant": response, "emotion": emotion})
            memory = load_memory()
            memory["timeline"].append({"text": user_input, "emotion": emotion, "time": str(datetime.now())})
            save_memory(memory)

    for chat in st.session_state.chat_history[::-1]:
        st.markdown(f"**🧍 You:** {chat['user']}")
        st.markdown(f"**🤖 EchoSoul:** {chat['assistant']} *(you seemed {chat['emotion']})*")
        st.markdown("---")

# Mode: Chat History
elif mode == "Chat history":
    st.title("📜 Chat History")
    for chat in st.session_state.get("chat_history", []):
        st.markdown(f"🧍 {chat['user']} → 🤖 {chat['assistant']} ({chat['emotion']})")

# Mode: Life Timeline
elif mode == "Life timeline":
    st.title("🕰️ Life Timeline")
    memory = load_memory()
    for event in memory["timeline"]:
        st.markdown(f"{event['time']}: **{event['text']}** — _{event['emotion']}_")

# Mode: Vault
elif mode == "Vault":
    st.title("🔒 Personal Vault")
    memory = load_memory()
    new_key = st.text_input("Key")
    new_value = st.text_input("Value")
    if st.button("Save"):
        memory["vault"][new_key] = new_value
        save_memory(memory)
        st.success("Saved successfully!")
    st.json(memory["vault"])

# Mode: Call
elif mode == "Call":
    st.title("📞 Live Voice Chat")

    class EchoProcessor(AudioProcessorBase):
        def recv_audio(self, frame):
            return frame

    webrtc_streamer(
        key="call",
        mode=WebRtcMode.SENDRECV,
        audio_processor_factory=EchoProcessor,
        media_stream_constraints={"audio": True, "video": False}
    )

# Mode: About
elif mode == "About":
    st.title("💠 About EchoSoul")
    st.markdown("""
    **EchoSoul** is your emotional AI companion — capable of:
    - Remembering personal details (Persistent Memory)
    - Adapting tone dynamically (Adaptive Personality)
    - Detecting mood via NLP (Emotion Recognition)
    - Building your Life Timeline 🕰️
    - Secure Vault 🔒
    - Live Audio Call Support 🎧
    """)
