# app.py
"""
EchoSoul - Streamlit single-file app
Features:
- Persistent encrypted memory (SQLite + Fernet)
- Adaptive personality (tone + persona)
- NLP: TextBlob sentiment + optional spaCy (NER & noun-chunks)
- OpenAI LLM for adaptive responses (OPENAI_API_KEY)
- Conversation, Memory Vault, Life Timeline, Time-Shifted Self, Legacy Mode, Soul Resonance
- Live Call (simulated) via streamlit-webrtc (optional)
- Chat history stored in session_state so replies persist; chatbox clears after send
"""

import os
import json
import sqlite3
import traceback
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st
from textblob import TextBlob
import openai  # ✅ fixed import

# Optional spaCy (if available)
try:
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # model may be missing at runtime; set nlp to None gracefully
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

openai.api_key = OPENAI_API_KEY

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
        resp = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, max_tokens=max_tokens, temperature=0.8)
        return True, resp.choices[0].message["content"].strip()
    except Exception as e:  # ✅ fixed exception handling
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
        # Fallback
        analysis = analyze_text_nlp(user_text)
        fallback = f"[{profile.get('tone','warm')} voice | mood: {analysis['mood']}] I hear you: \"{user_text}\"."
        return False, f"{resp}\n\nFallback: {fallback}"


# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul — adaptive, memoryful AI companion")

# Session state init
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []  # list of messages: {"role":"user|echo|system", "text":..., "time":...}

if "profile" not in st.session_state:
    st.session_state["profile"] = {"name": "", "tone": "warm", "persona": "A supportive companion."}

# Sidebar controls
with st.sidebar:
    st.header("EchoSoul Controls")
    st.session_state.profile["name"] = st.text_input("Your name", value=st.session_state.profile.get("name", ""))
    st.session_state.profile["tone"] = st.selectbox(
        "Personality tone", ["warm", "scientific", "playful", "stoic", "empathetic"], index=["warm", "scientific", "playful", "stoic", "empathetic"].index(st.session_state.profile.get("tone", "warm"))
    )
    st.session_state.profile["persona"] = st.text_area("Persona description (how EchoSoul sounds)", value=st.session_state.profile.get("persona", "A supportive companion."))
    if st.button("Save profile"):
        st.success("Profile saved")

    st.markdown("---")
    st.markdown("**Encryption / Vault**")
    if cipher:
        st.success("Vault enabled — encrypted memories will be saved.")
    else:
        st.warning("ECHOSOUL_KEY not set or invalid. Encrypted vault disabled.")
        st.caption("Set environment var ECHOSOUL_KEY to enable encryption (Fernet key).")

    st.markdown("---")
    st.caption("Set OPENAI_API_KEY in environment to enable LLM responses.")

# Tabs
tabs = st.tabs(["Conversation", "Memory Vault", "Life Timeline", "Time-Shifted Self", "Legacy Mode", "Soul Resonance", "Live Call (Sim)"])

# ---------- Conversation ----------
with tabs[0]:
    st.header("Talk to EchoSoul")
    if "chat_input" not in st.session_state:
        st.session_state["chat_input"] = ""

    user_text = st.text_area("Say something", value=st.session_state["chat_input"], key="chat_input_box", height=120)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        save_enc = st.checkbox("Save to memory (encrypted)", value=False, key="save_enc")
    with col2:
        mem_title = st.text_input("Memory title", value=f"Memory {datetime.utcnow().date()}", key="mem_title")
    with col3:
        st.write("")  # spacer
        send = st.button("Send to EchoSoul", type="primary")

    if send:
        if user_text.strip():
            st.session_state["chat_input"] = user_text
            analysis = analyze_text_nlp(user_text)

            # build context from recent memories
            recent = list_memories_db(limit=6)
            ctx = ""
            for m in recent:
                mem = get_memory_db(m["id"])
                if mem:
                    ctx += f"{mem['title']}: {mem['content']}\n"

            ok, reply = generate_echo_response(user_text, st.session_state["profile"], context=ctx, mode="reflect")

            st.session_state["chat_history"].append({"role": "user", "text": user_text, "time": datetime.utcnow().isoformat()})
            st.session_state["chat_history"].append({"role": "echo", "text": reply, "time": datetime.utcnow().isoformat(), "ok": ok})

            if save_enc:
                try:
                    to_store = cipher.encrypt(user_text.encode()).decode() if cipher else user_text
                    meta = {"analysis": analysis, "source": "conversation"}
                    save_memory_db(mem_title, to_store, encrypted=bool(cipher), metadata=meta)
                    st.session_state["chat_history"].append({"role": "system", "text": f"Memory '{mem_title}' saved.", "time": datetime.utcnow().isoformat()})
                except Exception as e:
                    st.session_state["chat_history"].append({"role": "system", "text": f"Save error: {e}", "time": datetime.utcnow().isoformat()})

            # Clear input and rerun so widget resets visually
            st.session_state["chat_input"] = ""
            st.rerun()

    st.markdown("---")
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

# ---------- Memory Vault ----------
with tabs[1]:
    st.header("Memory Vault")
    st.write("Securely stored memories. Decrypted only if ECHOSOUL_KEY is configured.")
    mems = list_memories_db(limit=200)
    if not mems:
        st.info("No memories yet.")
    else:
        for m in mems:
            with st.expander(f"{m['title']} — {m['created_at'][:19]}"):
                mem = get_memory_db(m["id"])
                if mem:
                    st.write(mem["content"])
                    st.json(mem["metadata"])
                else:
                    st.write("Error reading memory.")

    st.markdown("---")
    if st.button("Clear all memories (dangerous)", key="clear_all"):
        cur.execute("DELETE FROM memories")
        conn.commit()
        st.success("All memories cleared.")
        st.rerun()

# ---------- Life Timeline ----------
with tabs[2]:
    st.header("Life Timeline")
    cur.execute("SELECT id, created_at, title, encrypted FROM memories ORDER BY created_at ASC")
    rows = cur.fetchall()
    if not rows:
        st.info("No timeline entries yet.")
    else:
        for mid, created_at, title, enc in rows:
            mem = get_memory_db(mid)
            content = mem["content"] if mem else ""
            st.markdown(f"**{created_at[:10]} — {title}**")
            st.write(content)

# ---------- Time-Shifted Self ----------
with tabs[3]:
    st.header("Time-Shifted Self")
    ts_mode = st.radio("Mode", ["past", "future", "reflection"])
    ts_years = st.slider("Years shift", min_value=1, max_value=50, value=5)
    ts_prompt = st.text_area("What would you ask your time-shifted self?", height=120)
    if st.button("Talk to time-shifted self", key="timeshift"):
        profile = dict(st.session_state["profile"])
        profile["persona"] = f"{profile.get('persona','')} (as your {ts_mode} self, {ts_years} years {'ago' if ts_mode=='past' else 'ahead'})"
        recent = list_memories_db(limit=10)
        ctx = "\n".join([get_memory_db(m["id"])["content"] for m in recent if get_memory_db(m["id"])])
        ok, reply = generate_echo_response(ts_prompt or "Hello", profile, context=ctx, mode="time-shift")
        st.markdown(f"**Time-Shifted Echo ({ts_mode}, {ts_years}y):** {reply}")
        st.session_state["chat_history"].append({"role": "echo", "text": f"(Time-shifted) {reply}", "time": datetime.utcnow().isoformat(), "ok": ok})

# ---------- Legacy Mode ----------
with tabs[4]:
    st.header("Legacy Mode")
    if "legacy_q" not in st.session_state:
        st.session_state["legacy_q"] = ""
    legacy_q = st.text_input("Ask your legacy self", value=st.session_state["legacy_q"], key="legacy_box")
    if st.button("Consult Legacy Soul"):
        if legacy_q.strip():
            all_mem = list_memories_db(limit=50)
            ctx = "\n".join([get_memory_db(m["id"])["content"] for m in all_mem if get_memory_db(m["id"])])
            ok, reply = generate_echo_response(legacy_q, st.session_state["profile"], context=ctx, mode="legacy")
            st.markdown(f"**Legacy Echo:** {reply}")
            st.session_state["chat_history"].append({"role": "echo", "text": f"(Legacy) {reply}", "time": datetime.utcnow().isoformat(), "ok": ok})
            st.session_state["legacy_q"] = ""
            st.rerun()

# ---------- Soul Resonance ----------
with tabs[5]:
    st.header("Soul Resonance Network")
    if "res_input" not in st.session_state:
        st.session_state["res_input"] = ""
    res_in = st.text_input("Send to Soul Resonance", value=st.session_state["res_input"], key="res_box")
    if st.button("Send to Resonance"):
        if res_in.strip():
            ok, reply = generate_echo_response(res_in, st.session_state["profile"], context="Resonance network collectives", mode="resonance")
            st.markdown(f"**Resonance Reply:** {reply}")
            st.session_state["chat_history"].append({"role": "echo", "text": f"(Resonance) {reply}", "time": datetime.utcnow().isoformat(), "ok": ok})
            st.session_state["res_input"] = ""
            st.rerun()

# ---------- Live Call (Sim) ----------
with tabs[6]:
    st.header("Live Call (Simulated)")
    st.write("This demo captures audio (if environment supports streamlit-webrtc + ffmpeg). Replace with Twilio/Agora for production PSTN/VoIP.")
    if not STREAMLIT_WEBRTC_AVAILABLE:
        st.warning("streamlit-webrtc not available. Live call disabled.")
    else:
        class SimpleAudioProcessor(AudioProcessorBase):
            def recv(self, frame):
                # placeholder for audio processing / emotion detection
                return frame

        webrtc_streamer(key="echosoul_live", audio_processor_factory=SimpleAudioProcessor)
        st.caption("Captured audio frames are sent to the placeholder processor.")

st.markdown("---")
st.caption("EchoSoul — scaffold for the features listed. Replace or extend LLM/provider and voice/emotion models for production.")
