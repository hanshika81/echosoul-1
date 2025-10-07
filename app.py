"""
EchoSoul - Streamlit single-file app
Now with persistent chat history stored in SQLite
"""

import os
import json
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any

import streamlit as st
from textblob import TextBlob
from openai import OpenAI   # ✅ fixed import

# Optional spaCy
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except Exception:
    nlp = None

# Optional WebRTC
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
    STREAMLIT_WEBRTC_AVAILABLE = True
except Exception:
    STREAMLIT_WEBRTC_AVAILABLE = False

# ---------------------- Config ----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ECHOSOUL_KEY = os.getenv("ECHOSOUL_KEY")  # Fernet key
DB_PATH = os.getenv("ECHOSOUL_DB", "echosoul.db")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ---------------------- Encryption ----------------------
def get_cipher():
    if not ECHOSOUL_KEY:
        return None
    try:
        from cryptography.fernet import Fernet
        return Fernet(ECHOSOUL_KEY.encode() if isinstance(ECHOSOUL_KEY, str) else ECHOSOUL_KEY)
    except Exception:
        return None

cipher = get_cipher()

# ---------------------- Database ----------------------
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path, check_same_thread=False)
    cur = conn.cursor()
    # Memories
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
    # Chat history
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            role TEXT,
            text TEXT,
            ok INTEGER DEFAULT 1,
            metadata TEXT
        )
        """
    )
    conn.commit()
    return conn

conn = init_db()
cur = conn.cursor()

# ---- Memory funcs ----
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
    return [{"id": r[0], "created_at": r[1], "title": r[2], "encrypted": bool(r[3]), "metadata": json.loads(r[4] or "{}")} for r in rows]

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

# ---- Chat funcs ----
def save_chat_message(role: str, text: str, ok: bool = True, metadata: Optional[dict] = None):
    cur.execute(
        "INSERT INTO chat_history (created_at, role, text, ok, metadata) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(), role, text, 1 if ok else 0, json.dumps(metadata or {}))
    )
    conn.commit()

def list_chat_history(limit: int = 200):
    cur.execute("SELECT created_at, role, text, ok FROM chat_history ORDER BY id ASC LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"time": r[0], "role": r[1], "text": r[2], "ok": bool(r[3])} for r in rows]

# ---------------------- NLP ----------------------
def analyze_text_nlp(text: str) -> Dict[str, Any]:
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
    except Exception:
        polarity = 0.0
    if polarity > 0.4: mood = "happy"
    elif polarity > 0.05: mood = "calm"
    elif polarity < -0.4: mood = "angry/sad"
    elif polarity < -0.05: mood = "upset"
    else: mood = "neutral"
    entities, key_phrases = [], []
    if nlp:
        try:
            doc = nlp(text)
            entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents]
            key_phrases = [chunk.text for chunk in doc.noun_chunks]
        except Exception:
            pass
    return {"polarity": polarity, "mood": mood, "entities": entities, "key_phrases": key_phrases}

# ---------------------- LLM ----------------------
def call_openai_chat(messages: list, max_tokens: int = 300):
    if not OPENAI_API_KEY:
        return False, "OpenAI API key not configured."
    try:
        resp = client.chat.completions.create(
            model="openrouter/auto",
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.8
        )
        return True, resp.choices[0].message.content.strip()
    except Exception as e:
        return False, f"OpenRouter error: {str(e)}"

def generate_echo_response(user_text: str, profile: dict, context: Optional[str] = None, mode: str = "reflect"):
    prefix = f"You are EchoSoul — an adaptive, compassionate assistant.\nTone: {profile.get('tone','warm')}\nPersona: {profile.get('persona','A supportive companion')}."
    if profile.get("name"):
        prefix += f" Address the user by name: {profile.get('name')}."
    if mode == "legacy":
        prefix += " Answer like a wise older version of the user."
    if mode == "time-shift":
        prefix += " Answer as a time-shifted version of the user."
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

# ---------------------- UI ----------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul — adaptive, memoryful AI companion")

if "profile" not in st.session_state:
    st.session_state["profile"] = {"name": "", "tone": "warm", "persona": "A supportive companion."}

with st.sidebar:
    st.header("EchoSoul Controls")
    st.session_state.profile["name"] = st.text_input("Your name", value=st.session_state.profile.get("name", ""))
    st.session_state.profile["tone"] = st.selectbox("Personality tone", ["warm","scientific","playful","stoic","empathetic"],
        index=["warm","scientific","playful","stoic","empathetic"].index(st.session_state.profile.get("tone","warm")))
    st.session_state.profile["persona"] = st.text_area("Persona description", value=st.session_state.profile.get("persona","A supportive companion."))
    if st.button("Save profile"): st.success("Profile saved")
    st.markdown("---")
    if cipher: st.success("Vault enabled — encrypted memories will be saved.")
    else: st.warning("Vault disabled. Set ECHOSOUL_KEY env.")
    st.markdown("---")
    st.caption("Set OPENAI_API_KEY to enable LLM responses.")

tabs = st.tabs(["Conversation","Memory Vault","Life Timeline","Time-Shifted Self","Legacy Mode","Soul Resonance","Live Call (Sim)"])

# ---------- Conversation ----------
with tabs[0]:
    st.header("Talk to EchoSoul")
    user_text = st.text_area("Say something", height=120, key="chat_input_box")
    col1, col2, col3 = st.columns([1,1,1])
    with col1: save_enc = st.checkbox("Save to memory (encrypted)", value=False, key="save_enc")
    with col2: mem_title = st.text_input("Memory title", value=f"Memory {datetime.utcnow().date()}", key="mem_title")
    with col3: send = st.button("Send to EchoSoul", type="primary")
    if send and user_text.strip():
        analysis = analyze_text_nlp(user_text)
        recent = list_memories_db(limit=6)
        ctx = "\n".join([get_memory_db(m["id"])["content"] for m in recent if get_memory_db(m["id"])])
        ok, reply = generate_echo_response(user_text, st.session_state["profile"], context=ctx, mode="reflect")
        save_chat_message("user", user_text, True)
        save_chat_message("echo", reply, ok)
        if save_enc:
            to_store = cipher.encrypt(user_text.encode()).decode() if cipher else user_text
            meta = {"analysis": analysis, "source": "conversation"}
            save_memory_db(mem_title, to_store, encrypted=bool(cipher), metadata=meta)
            save_chat_message("system", f"Memory '{mem_title}' saved.")
        st.rerun()
    st.markdown("#### Conversation history")
    for msg in list_chat_history(limit=200):
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        elif msg["role"] == "echo":
            st.markdown(f"**EchoSoul:** {msg['text']}" if msg["ok"] else f"**EchoSoul (error/fallback):** {msg['text']}")
        else:
            st.info(msg["text"])

# ---------- Memory Vault ----------
with tabs[1]:
    st.header("Memory Vault")
    mems = list_memories_db(limit=200)
    if not mems: st.info("No memories yet.")
    else:
        for m in mems:
            with st.expander(f"{m['title']} — {m['created_at'][:19]}"):
                mem = get_memory_db(m["id"])
                if mem: st.write(mem["content"]); st.json(mem["metadata"])
                else: st.write("Error reading memory.")
    if st.button("Clear all memories (dangerous)", key="clear_all"):
        cur.execute("DELETE FROM memories"); conn.commit(); st.success("All memories cleared."); st.rerun()

# ---------- Life Timeline ----------
with tabs[2]:
    st.header("Life Timeline")
    cur.execute("SELECT id, created_at, title FROM memories ORDER BY created_at ASC")
    rows = cur.fetchall()
    if not rows: st.info("No timeline entries yet.")
    else:
        for mid, created_at, title in rows:
            mem = get_memory_db(mid)
            st.markdown(f"{created_at[:10]} — {title}")
            st.write(mem["content"] if mem else "")

# ---------- Time-Shifted Self ----------
with tabs[3]:
    st.header("Time-Shifted Self")
    ts_mode = st.radio("Mode", ["past","future","reflection"])
    ts_years = st.slider("Years shift", 1, 50, 5)
    ts_prompt = st.text_area("What would you ask your time-shifted self?", height=120)
    if st.button("Talk to time-shifted self", key="timeshift"):
        profile = dict(st.session_state["profile"])
        profile["persona"] += f" (as your {ts_mode} self, {ts_years} years {'ago' if ts_mode=='past' else 'ahead'})"
        recent = list_memories_db(limit=10)
        ctx = "\n".join([get_memory_db(m["id"])["content"] for m in recent if get_memory_db(m["id"])])
        ok, reply = generate_echo_response(ts_prompt or "Hello", profile, context=ctx, mode="time-shift")
        save_chat_message("user", ts_prompt or "Hello")
        save_chat_message("echo", f"(Time-shifted) {reply}", ok)
        st.markdown(f"Time-Shifted Echo ({ts_mode}, {ts_years}y): {reply}")

# ---------- Legacy Mode ----------
with tabs[4]:
    st.header("Legacy Mode")
    legacy_q = st.text_input("Ask your legacy self", key="legacy_box")
    if st.button("Consult Legacy Soul"):
        if legacy_q.strip():
            all_mem = list_memories_db(limit=50)
            ctx = "\n".join([get_memory_db(m["id"])["content"] for m in all_mem if get_memory_db(m["id"])])
            ok, reply = generate_echo_response(legacy_q, st.session_state["profile"], context=ctx, mode="legacy")
            save_chat_message("user", legacy_q)
            save_chat_message("echo", f"(Legacy) {reply}", ok)
            st.markdown(f"Legacy Echo: {reply}")

# ---------- Soul Resonance ----------
with tabs[5]:
    st.header("Soul Resonance Network")
    res_in = st.text_input("Send to Soul Resonance", key="res_box")
    if st.button("Send to Resonance"):
        if res_in.strip():
            ok, reply = generate_echo_response(res_in, st.session_state["profile"], context="Resonance network", mode="resonance")
            save_chat_message("user", res_in)
            save_chat_message("echo", f"(Resonance) {reply}", ok)
            st.markdown(f"Resonance Reply: {reply}")

# ---------- Live Call ----------
with tabs[6]:
    st.header("Live Call (Simulated)")
    if not STREAMLIT_WEBRTC_AVAILABLE:
        st.warning("streamlit-webrtc not available. Live call disabled.")
    else:
        class SimpleAudioProcessor(AudioProcessorBase):
            def recv(self, frame): return frame
        webrtc_streamer(key="echosoul_live", audio_processor_factory=SimpleAudioProcessor)
        st.caption("Captured audio frames are sent to placeholder processor.")

st.caption("EchoSoul — scaffold for the features listed. Chat + memories now persist in DB.")
