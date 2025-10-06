# app.py
import os
import json
import sqlite3
import streamlit as st
from datetime import datetime
from cryptography.fernet import Fernet, InvalidToken
from textblob import TextBlob

# NLP (spaCy)
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download as spacy_download
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
except Exception as e:
    nlp = None
    print("spaCy not available:", e)

# OpenAI
from openai import OpenAI
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# streamlit-webrtc (optional)
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
    STREAMLIT_WEBRTC_AVAILABLE = True
except Exception:
    STREAMLIT_WEBRTC_AVAILABLE = False

# ---- Configuration ----
DB_PATH = os.environ.get("ECHOSOUL_DB", "echosoul.db")
FERNET_KEY = os.environ.get("ECHOSOUL_KEY")  # set in Render env

# ---- DB setup ----
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        title TEXT,
        content BLOB,
        encrypted INTEGER DEFAULT 0,
        metadata TEXT
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS profile (
        id INTEGER PRIMARY KEY,
        data TEXT
    )""")
    conn.commit()
    return conn

conn = init_db()

# ---- Encryption ----
def get_fernet():
    if not FERNET_KEY:
        return None
    key = FERNET_KEY.encode() if isinstance(FERNET_KEY, str) else FERNET_KEY
    return Fernet(key)

def encrypt_text(plain: str) -> bytes:
    f = get_fernet()
    if not f:
        raise RuntimeError("Encryption key not configured.")
    return f.encrypt(plain.encode('utf-8'))

def decrypt_bytes(blob: bytes) -> str:
    f = get_fernet()
    if not f:
        raise RuntimeError("Encryption key not configured.")
    if isinstance(blob, str):
        blob = blob.encode('latin1')
    return f.decrypt(blob).decode('utf-8')

# ---- Memory management ----
def save_memory(title: str, content: str, encrypt: bool=False, metadata: dict=None):
    cur = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    metadata_json = json.dumps(metadata or {})
    if encrypt:
        blob = encrypt_text(content)
        cur.execute(
            "INSERT INTO memories (created_at, title, content, encrypted, metadata) VALUES (?, ?, ?, ?, ?)",
            (created_at, title, blob, 1, metadata_json)
        )
    else:
        cur.execute(
            "INSERT INTO memories (created_at, title, content, encrypted, metadata) VALUES (?, ?, ?, ?, ?)",
            (created_at, title, content, 0, metadata_json)
        )
    conn.commit()

def list_memories():
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, title, encrypted, metadata FROM memories ORDER BY created_at DESC")
    rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "created_at": r[1],
            "title": r[2],
            "encrypted": bool(r[3]),
            "metadata": json.loads(r[4] or "{}")
        }
        for r in rows
    ]

def get_memory(mid: int):
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, title, content, encrypted, metadata FROM memories WHERE id = ?", (mid,))
    r = cur.fetchone()
    if not r:
        return None
    content = r[3]
    if r[4]:
        try:
            content = decrypt_bytes(content)
        except Exception as e:
            content = f"<ERROR DECRYPTING: {e}>"
    return {
        "id": r[0],
        "created_at": r[1],
        "title": r[2],
        "content": content,
        "encrypted": bool(r[4]),
        "metadata": json.loads(r[5] or "{}")
    }

# ---- Profile ----
def save_profile(profile: dict):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO profile (id, data) VALUES (1, ?)", (json.dumps(profile),))
    conn.commit()

def load_profile():
    cur = conn.cursor()
    cur.execute("SELECT data FROM profile WHERE id = 1")
    r = cur.fetchone()
    return json.loads(r[0]) if r else {}

# ---- NLP ----
def analyze_text_nlp(text: str) -> dict:
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
    except Exception:
        polarity = 0.0

    mood = "neutral"
    if polarity > 0.4: mood = "happy"
    elif polarity > 0.05: mood = "calm"
    elif polarity < -0.4: mood = "angry/sad"
    elif polarity < -0.05: mood = "upset/negative"

    entities, key_phrases = [], []
    if nlp:
        try:
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            key_phrases = [chunk.text for chunk in doc.noun_chunks]
        except: pass

    return {"polarity": polarity, "mood": mood, "entities": entities, "key_phrases": key_phrases}

# ---- Generate response with OpenAI ----
def generate_echo_response(user_text: str, profile: dict):
    analysis = analyze_text_nlp(user_text)
    tone = profile.get("tone", "warm")
    persona = profile.get("persona_description", "A supportive companion.")
    mood = analysis.get("mood", "neutral")

    if client:
        try:
            system_prompt = f"You are EchoSoul, an adaptive AI companion. Tone: {tone}. Persona: {persona}. The user’s current mood is {mood}. Respond with empathy and adapt to their style."
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                max_tokens=250
            )
            response = resp.choices[0].message.content
        except Exception as e:
            response = f"(OpenAI error: {e})"
    else:
        response = f"[{tone} voice | mood: {mood}] I hear you. You said: \"{user_text}\"."

    return response, analysis

# ---- Streamlit UI ----
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul — personal adaptive companion (demo)")

with st.sidebar:
    st.header("EchoSoul Controls")
    profile = load_profile()
    name = st.text_input("Your name", value=profile.get("name",""))
    tone = st.selectbox("Personality tone", ["warm","scientific","playful","stoic","empathetic"], index=0)
    persona = st.text_area("Persona description", value=profile.get("persona_description","A supportive companion."))
    if st.button("Save profile"):
        save_profile({"name": name, "tone": tone, "persona_description": persona})
        st.success("Saved profile")

    st.markdown("---")
    st.write("Encryption:")
    if FERNET_KEY:
        st.success("Encryption enabled")
    else:
        st.warning("ECHOSOUL_KEY not set. Vault disabled.")
    st.markdown("---")
    st.caption("Set `ECHOSOUL_KEY` and `OPENAI_API_KEY` in Render environment.")

tabs = st.tabs(["Conversation", "Memory Vault", "Life Timeline", "Time-Shifted Self", "Live Call (Sim)"])

# Conversation
with tabs[0]:
    st.header("Talk to EchoSoul")
    profile = load_profile()
    user_input = st.text_area("Say something", height=120)
    encrypt_mem = st.checkbox("Save to memory (encrypted)", value=False)
    title = st.text_input("Memory title", value=f"Memory {datetime.utcnow().date()}")
    if st.button("Send to EchoSoul"):
        if not user_input.strip():
            st.warning("Write something first.")
        else:
            resp, analysis = generate_echo_response(user_input, profile)
            st.markdown("**EchoSoul:**")
            st.write(resp)
            try:
                save_memory(title, user_input, encrypt=encrypt_mem, metadata={"analysis": analysis})
            except Exception as e:
                st.error(f"Save error: {e}")

# Memory Vault
with tabs[1]:
    st.header("Memory Vault")
    mems = list_memories()
    for m in mems:
        if st.button(f"{m['id']}: {m['title']} ({m['created_at'][:19]})"):
            mem = get_memory(m['id'])
            st.subheader(mem["title"])
            st.write(mem["created_at"])
            st.write(mem["content"])
            st.json(mem["metadata"])

# Life Timeline
with tabs[2]:
    st.header("Life Timeline")
    for m in list_memories():
        st.write(f"**{m['title']}** — {m['created_at'][:19]}")

# Time-Shifted Self
with tabs[3]:
    st.header("Time-Shifted Self")
    mode = st.radio("Mode", ["past","future"])
    years = st.slider("Years shift",1,50,5)
    prompt = st.text_area("Message")
    if st.button("Talk to time-shifted self"):
        temp_profile = dict(profile)
        temp_profile["persona_description"] = f"{profile.get('persona_description','')} (as your {mode} self {years} years {'ago' if mode=='past' else 'ahead'})"
        resp, _ = generate_echo_response(prompt, temp_profile)
        st.write(resp)

# Live Call (Sim)
with tabs[4]:
    st.header("Live Call (Sim)")
    if not STREAMLIT_WEBRTC_AVAILABLE:
        st.warning("Install `streamlit-webrtc` + ffmpeg to enable.")
    else:
        class SimpleAudioProcessor(AudioProcessorBase):
            def recv(self, frame): return frame
        webrtc_streamer(key="call", audio_processor_factory=SimpleAudioProcessor)
        st.info("Audio captured (placeholder). Replace with Twilio/Agora for real calls.")
