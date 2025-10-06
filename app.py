# app.py
"""
EchoSoul - Streamlit single-file app
...
"""

import os
import json
import sqlite3
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st
from textblob import TextBlob
from openai import OpenAI   # ✅ fixed import (new API)

# Optional spaCy (if available)
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None

# Optional WebRTC (live call simulation)
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
    STREAMLIT_WEBRTC_AVAILABLE = True
except Exception:
    STREAMLIT_WEBRTC_AVAILABLE = False

# ----------------------
# Config & environment
# ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ECHOSOUL_KEY = os.getenv("ECHOSOUL_KEY")  # Fernet key (base64)
DB_PATH = os.getenv("ECHOSOUL_DB", "echosoul.db")

# ✅ OpenRouter client setup
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

# ----------------------
# Encryption helper (Fernet)
# ----------------------
def get_cipher():
    if not ECHOSOUL_KEY:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(ECHOSOUL_KEY.encode() if isinstance(ECHOSOUL_KEY, str) else ECHOSOUL_KEY)
    except Exception:
        return None

cipher = get_cipher()

# ----------------------
# Database setup
# ----------------------
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            title TEXT,
            content TEXT,
            encrypted INTEGER DEFAULT 0,
            metadata TEXT
        )
        """
    )
    conn.commit()
    return conn

conn = init_db()
cur = conn.cursor()

def save_memory_db(title: str, content: str, encrypted: bool, metadata: Optional[dict] = None):
    meta = json.dumps(metadata or {})
    enc_flag = 1 if encrypted else 0
    cur.execute(
        "INSERT INTO memories (created_at, title, content, encrypted, metadata) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), title, content, enc_flag, meta),
    )
    conn.commit()
    return cur.lastrowid

def list_memories_db(limit: int = 200):
    cur.execute("SELECT id, created_at, title, encrypted, metadata FROM memories ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    out = []
    for r in rows:
        out.append({"id": r[0], "created_at": r[1], "title": r[2], "encrypted": bool(r[3]), "metadata": json.loads(r[4] or "{}")})
    return out

def get_memory_db(mid: int):
    cur.execute("SELECT id, created_at, title, content, encrypted, metadata FROM memories WHERE id = ?", (mid,))
    r = cur.fetchone()
    if not r:
        return None
    content = r[3]
    if r[4]:
        if cipher:
            try:
                content = cipher.decrypt(content.encode()).decode()
            except Exception:
                content = "<DECRYPTION_FAILED>"
        else:
            content = "<ENCRYPTED - KEY NOT CONFIGURED>"
    return {"id": r[0], "created_at": r[1], "title": r[2], "content": content, "encrypted": bool(r[4]), "metadata": json.loads(r[5] or "{}")}

# ----------------------
# NLP helpers
# ----------------------
def analyze_text_nlp(text: str) -> Dict[str, Any]:
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
    except Exception:
        polarity = 0.0
    if polarity > 0.4:
        mood = "happy"
    elif polarity > 0.05:
        mood = "calm"
    elif polarity < -0.4:
        mood = "angry/sad"
    elif polarity < -0.05:
        mood = "upset"
    else:
        mood = "neutral"
    entities = []
    key_phrases = []
    if nlp:
        try:
            doc = nlp(text)
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
            key_phrases = [chunk.text for chunk in doc.noun_chunks]
        except Exception:
            entities = []
            key_phrases = []
    return {"polarity": polarity, "mood": mood, "entities": entities, "key_phrases": key_phrases}

# ----------------------
# OpenAI wrapper
# ----------------------
def call_openai_chat(messages: list, max_tokens: int = 300):
    if not OPENAI_API_KEY:
        return False, "OpenAI API key not configured (OPENAI_API_KEY)."
    try:
        resp = client.chat.completions.create(   # ✅ new API
            model="openrouter/auto",             # ✅ OpenRouter auto model
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.8
        )
        return True, resp.choices[0].message.content.strip()
    except Exception as e:
        return False, f"OpenAI error: {str(e)}"

def generate_echo_response(user_text: str, profile: dict, context: Optional[str] = None, mode: str = "reflect"):
    prefix = f"You are EchoSoul — an adaptive, compassionate assistant.\nTone: {profile.get('tone','warm')}\nPersona: {profile.get('persona','A supportive companion')}."
    if profile.get("name"):
        prefix += f" Address the user by name: {profile.get('name')}."
    if mode == "legacy":
        prefix += " Answer like a wise older version of the user."
    if context:
        prefix += f"\nContext (recent memories):\n{context}"
    system_msg = {"role": "system", "content": prefix}
    user_msg = {"role": "user", "content": user_text}
    ok, resp = call_openai_chat([system_msg, user_msg], max_tokens=400)
    if ok:
        return True, resp
    else:
        analysis = analyze_text_nlp(user_text)
        fallback = f"[{profile.get('tone','warm')} voice | mood: {analysis['mood']}] I hear you: \"{user_text}\"."
        return False, f"{resp}\n\nFallback: {fallback}"

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul — adaptive, memoryful AI companion")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "profile" not in st.session_state:
    st.session_state["profile"] = {"name": "", "tone": "warm", "persona": "A supportive companion."}

# Sidebar controls (unchanged)...

# ----------------------
# Conversation Tab Fix
# ----------------------
with st.tabs(["Conversation", "Memory Vault", "Life Timeline", "Time-Shifted Self", "Legacy Mode", "Soul Resonance", "Live Call (Sim)"])[0]:
    st.header("Talk to EchoSoul")
    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = ""

    user_text = st.text_area("Say something", value=st.session_state["chat_input"], key="chat_input_box", height=120)
    send = st.button("Send to EchoSoul", type="primary")

    if send and user_text.strip():
        st.session_state["chat_input"] = ""  # ✅ clears box after send
        analysis = analyze_text_nlp(user_text)
        recent = list_memories_db(limit=6)
        ctx = "\n".join([get_memory_db(m["id"])["content"] for m in recent if get_memory_db(m["id"])])
        ok, reply = generate_echo_response(user_text, st.session_state["profile"], context=ctx, mode="reflect")
        st.session_state["chat_history"].append({"role": "user", "text": user_text, "time": datetime.utcnow().isoformat()})
        st.session_state["chat_history"].append({"role": "echo", "text": reply, "time": datetime.utcnow().isoformat(), "ok": ok})
        st.rerun()

    st.markdown("#### Conversation history")
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        elif msg["role"] == "echo":
            if msg.get("ok"):
                st.markdown(f"**EchoSoul:** {msg['text']}")
            else:
                st.markdown(f"**EchoSoul (error/fallback):** {msg['text']}")
        else:
            st.info(msg["text"])
