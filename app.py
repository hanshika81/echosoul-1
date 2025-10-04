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

# Try to import spaCy (optional). If available, will be used for keyword extraction.
try:
    import spacy
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

# -------------------------
# Load env & set config
# -------------------------
load_dotenv()  # local .env support for testing

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
AGORA_APP_ID = os.getenv("AGORA_APP_ID", "")
AGORA_APP_CERTIFICATE = os.getenv("AGORA_APP_CERTIFICATE", "")

openai.api_key = OPENAI_API_KEY or None

st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("🌙 EchoSoul — your adaptive AI companion")

# -------------------------
# ensure spaCy model available if spaCy installed
# -------------------------
if _HAS_SPACY:
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # try to download at runtime (Dockerfile will prefer to install it during build)
        try:
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None
            _HAS_SPACY = False
else:
    nlp = None

# -------------------------
# Data & persistent storage
# -------------------------
DATA_DIR = "data"
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
VAULT_FILE = os.path.join(DATA_DIR, "vault.enc")

os.makedirs(DATA_DIR, exist_ok=True)

def _load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def _save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

memory_store = _load_json(MEMORY_FILE, {"messages": [], "timeline": [], "personality": "neutral"})

def persist_memory():
    _save_json(MEMORY_FILE, memory_store)

# -------------------------
# NLP helpers (TextBlob primary, spaCy optional)
# -------------------------
def analyze_sentiment(text: str) -> Tuple[str, float]:
    tb = TextBlob(text)
    polarity = round(tb.sentiment.polarity, 3)  # -1..1
    if polarity > 0.15:
        return "positive", polarity
    if polarity < -0.15:
        return "negative", polarity
    return "neutral", polarity

def extract_keywords(text: str, max_words: int = 6) -> List[str]:
    # Prefer spaCy nouns/adjectives if available
    if _HAS_SPACY and nlp is not None:
        doc = nlp(text)
        keywords = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            if token.pos_ in ("NOUN", "PROPN", "ADJ"):
                kw = token.lemma_.lower().strip()
                if len(kw) > 1 and kw not in keywords:
                    keywords.append(kw)
            if len(keywords) >= max_words:
                break
        if keywords:
            return keywords[:max_words]
    # Fallback: simple frequency-based selection with TextBlob
    tb = TextBlob(text)
    words = [w.lower().strip(".,!?;:") for w in tb.words if len(w) > 2]
    seen = []
    for w in words:
        if w not in seen:
            seen.append(w)
        if len(seen) >= max_words:
            break
    return seen[:max_words]

# -------------------------
# Vault (password -> deterministic Fernet key)
# -------------------------
def _derive_key_from_password(password: str) -> bytes:
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h)

def vault_save(text: str, password: str) -> None:
    key = _derive_key_from_password(password)
    f = Fernet(key)
    token = f.encrypt(text.encode("utf-8"))
    with open(VAULT_FILE, "wb") as f_out:
        f_out.write(token)

def vault_load(password: str) -> str:
    if not os.path.exists(VAULT_FILE):
        return ""
    key = _derive_key_from_password(password)
    f = Fernet(key)
    try:
        with open(VAULT_FILE, "rb") as f_in:
            token = f_in.read()
        return f.decrypt(token).decode("utf-8")
    except Exception:
        return ""

# -------------------------
# OpenAI helpers
# -------------------------
def openai_available() -> bool:
    return bool(openai.api_key)

def generate_chat_reply(messages: list, model: str = "gpt-3.5-turbo", max_tokens: int = 512) -> Tuple[str, str]:
    if not openai.api_key:
        return "", "OpenAI API key not set"
    try:
        response = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens)
        reply = response.choices[0].message["content"].strip()
        return reply, ""
    except Exception as e:
        return "", str(e)

# -------------------------
# Adaptive personality
# -------------------------
def update_personality_from_recent():
    msgs = [m for m in memory_store["messages"] if m["role"] == "user"]
    last = msgs[-10:]
    if not last:
        memory_store["personality"] = "neutral"
        persist_memory()
        return
    scores = []
    for m in last:
        _, polarity = analyze_sentiment(m["content"])
        scores.append(polarity)
    avg = sum(scores) / len(scores) if scores else 0.0
    if avg > 0.1:
        memory_store["personality"] = "positive"
    elif avg < -0.1:
        memory_store["personality"] = "reserved"
    else:
        memory_store["personality"] = "thoughtful"
    persist_memory()

# -------------------------
# UI + helpers
# -------------------------
def add_message(role: str, content: str):
    memory_store["messages"].append({"role": role, "content": content, "time": datetime.utcnow().isoformat()})
    persist_memory()

def add_timeline_event(text: str):
    memory_store["timeline"].append({"text": text, "time": datetime.utcnow().isoformat()})
    persist_memory()

class EchoAudioProcessor(AudioProcessorBase):
    def recv(self, frame):
        return frame

# -------------------------
# Streamlit UI
# -------------------------
st.sidebar.title("EchoSoul")
st.sidebar.markdown("Adaptive companion — chat, call, remember.")
if st.sidebar.checkbox("Show env debug (hide keys)", value=False):
    st.sidebar.write({"OPENAI_API_KEY": bool(OPENAI_API_KEY), "AGORA_APP_ID": bool(AGORA_APP_ID), "spaCy": _HAS_SPACY and (nlp is not None)})

tabs = st.tabs(["Chat", "Brain mimic", "Chat history", "Life timeline", "Vault", "Call", "Export", "About"])
tab_chat, tab_mimic, tab_history, tab_timeline, tab_vault, tab_call, tab_export, tab_about = tabs

# Chat tab
with tab_chat:
    st.header("💬 Chat with EchoSoul")
    update_personality_from_recent()
    st.info(f"Adaptive personality: **{memory_store.get('personality', 'neutral')}**")
    with st.form("chat_form", clear_on_submit=True):
        user_text = st.text_area("Your message", placeholder="Share your thoughts...")
        submit = st.form_submit_button("Send")
    if submit and user_text and user_text.strip():
        emotion_label, polarity = analyze_sentiment(user_text)
        keywords = extract_keywords(user_text)
        system_prompt = (
            f"You are EchoSoul, an adaptive AI companion. The user is feeling {emotion_label} (polarity={polarity}). "
            f"Stored personality: {memory_store.get('personality','neutral')}. Be concise and empathetic."
        )
        messages_for_openai = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_text}]
        reply, err = generate_chat_reply(messages_for_openai)
        if err:
            st.error(f"OpenAI error: {err}")
            fallback = f"I heard you: {user_text[:200]}... You seem {emotion_label}."
            st.warning("Using fallback responder.")
            st.markdown(f"**EchoSoul (fallback):** {fallback}")
            add_message("user", user_text)
            add_message("assistant", fallback)
            add_timeline_event(f"User wrote: '{user_text[:120]}...' (emotion={emotion_label})")
        else:
            st.markdown(f"**EchoSoul:** {reply}")
            add_message("user", user_text)
            add_message("assistant", reply)
            add_timeline_event(f"User said: '{user_text[:120]}...' (emotion={emotion_label})")
        update_personality_from_recent()

# Brain mimic tab
with tab_mimic:
    st.header("🧠 Brain Mimic")
    st.write("EchoSoul will mimic your recent style.")
    if st.button("Generate mimic response"):
        user_msgs = [m["content"] for m in memory_store["messages"] if m["role"] == "user"][-10:]
        if not user_msgs:
            st.info("Not enough messages yet.")
        else:
            sample = "\n\n".join(user_msgs[-5:])
            mimic_prompt = f"Mimic the user's style from these recent messages:\n\n{sample}"
            reply, err = generate_chat_reply([{"role": "system", "content": "You are a mimic."}, {"role": "user", "content": mimic_prompt}])
            if err:
                st.error(f"OpenAI error: {err}")
            else:
                st.success("Mimic generated:")
                st.write(reply)
                add_message("assistant_mimic", reply)

# Chat history tab
with tab_history:
    st.header("🗂 Chat history")
    if memory_store["messages"]:
        df = pd.DataFrame(memory_store["messages"])
        st.dataframe(df.sort_values("time", ascending=False).reset_index(drop=True))
    else:
        st.info("No messages yet.")

# Timeline tab
with tab_timeline:
    st.header("📜 Life Timeline")
    new_event = st.text_input("Add a timeline event")
    if st.button("Save event") and new_event.strip():
        add_timeline_event(new_event.strip())
        st.success("Event added.")
    if memory_store.get("timeline"):
        df = pd.DataFrame(memory_store["timeline"])
        st.table(df.sort_values("time", ascending=False).reset_index(drop=True))
    else:
        st.info("No timeline events yet.")

# Vault tab
with tab_vault:
    st.header("🔐 Private Vault")
    vault_pwd = st.text_input("Vault password", type="password", key="vault_pwd")
    col1, col2 = st.columns(2)
    with col1:
        secret_in = st.text_area("Write a private memory to save")
        if st.button("Save to Vault"):
            if not vault_pwd:
                st.error("Set a password first.")
            else:
                vault_save(secret_in or "", vault_pwd)
                st.success("Saved to vault (encrypted).")
    with col2:
        if st.button("Load from Vault"):
            if not vault_pwd:
                st.error("Enter vault password.")
            else:
                out = vault_load(vault_pwd)
                if out:
                    st.success("Decrypted vault content:")
                    st.text_area("Vault content", value=out, height=200)
                else:
                    st.error("Could not decrypt. Wrong password or no vault present.")

# Call tab
with tab_call:
    st.header("📞 Live Call (WebRTC / Agora-ready)")
    st.write("Allow microphone access in your browser. Set AGORA_APP_ID to enable Agora token flows.")
    if not AGORA_APP_ID:
        st.info("Agora App ID not set — WebRTC still works locally in browser.")
    else:
        st.success("Agora App ID configured.")
    if st.button("Start Call (ask for mic)"):
        try:
            webrtc_streamer(
                key="echosoul_call",
                mode=WebRtcMode.SENDRECV,
                audio_receiver_size=2048,
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                media_stream_constraints={"audio": True, "video": False},
                audio_processor_factory=EchoAudioProcessor,
                async_processing=True
            )
            st.success("WebRTC session started.")
        except Exception as e:
            st.error(f"Could not start WebRTC: {e}")
            st.info("Allow microphone access in the browser and try again.")

# Export tab
with tab_export:
    st.header("📤 Export / Import")
    if st.button("Download memory JSON"):
        st.download_button("Download memory.json", data=json.dumps(memory_store, indent=2), file_name="echosoul_memory.json")
    uploaded = st.file_uploader("Upload memory JSON to import (replaces current)", type=["json"])
    if uploaded:
        try:
            new = json.load(uploaded)
            if isinstance(new, dict):
                memory_store.clear()
                memory_store.update(new)
                persist_memory()
                st.success("Memory imported.")
            else:
                st.error("Invalid memory JSON.")
        except Exception as e:
            st.error(f"Import failed: {e}")

# About tab
with tab_about:
    st.header("ℹ️ About EchoSoul")
    st.markdown("""
    EchoSoul is an adaptive AI companion:
    - Persistent Memory
    - Adaptive Personality
    - Emotion Recognition (TextBlob + spaCy optional)
    - Life Timeline
    - Private Vault (encrypted)
    - Brain Mimic
    - WebRTC Voice Call (Agora-ready)
    """)
    st.markdown("Tips: If WebRTC can't access your mic, click the lock icon in the browser address bar → allow Microphone → reload.")
