import os
import sqlite3
import streamlit as st
from datetime import datetime
from cryptography.fernet import Fernet
from textblob import TextBlob
from openai import OpenAI

# --- Setup ---
st.set_page_config(page_title="EchoSoul AI", layout="wide")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ECHOSOUL_KEY = os.environ.get("ECHOSOUL_KEY", "")

client = OpenAI(api_key=OPENAI_API_KEY)

# Encryption
cipher = Fernet(ECHOSOUL_KEY.encode()) if ECHOSOUL_KEY else None

# SQLite memory database
conn = sqlite3.connect("echosoul.db", check_same_thread=False)
c = conn.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS memories
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT,
              content TEXT,
              encrypted INTEGER,
              timestamp TEXT)""")
conn.commit()

# --- Helper Functions ---
def encrypt_text(text):
    if cipher:
        return cipher.encrypt(text.encode()).decode()
    return text

def decrypt_text(text, encrypted):
    if encrypted and cipher:
        return cipher.decrypt(text.encode()).decode()
    return text

def detect_emotion(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "positive 🙂"
    elif polarity < -0.3:
        return "negative 🙁"
    return "neutral 😐"

def generate_echo_response(user_input, tone="neutral", persona="default"):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are EchoSoul, a {tone} and {persona} companion."},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"

# --- Sidebar Controls ---
st.sidebar.title("⚡ EchoSoul Settings")
mode = st.sidebar.radio("Choose a mode", ["Conversation", "Memory Vault", "Life Timeline", "Legacy Mode", "Soul Resonance"])
tone = st.sidebar.selectbox("Tone", ["neutral", "warm", "playful", "serious"])
persona = st.sidebar.text_input("Persona style (e.g. mentor, friend, philosopher)", "default")

if cipher:
    st.sidebar.success("🔐 Vault enabled")
else:
    st.sidebar.warning("🔑 ECHOSOUL_KEY not set. Vault disabled.")

# --- Conversation Mode ---
if mode == "Conversation":
    st.header("Talk to EchoSoul")

    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    user_input = st.text_input("Say something", value=st.session_state.chat_input, key="chat_input_box")
    save_memory = st.checkbox("Save to memory (encrypted)")
    memory_title = st.text_input("Memory title", f"Memory {datetime.today().date()}")

    if st.button("Send to EchoSoul"):
        if user_input.strip():
            emotion = detect_emotion(user_input)
            reply = generate_echo_response(user_input, tone, persona)

            st.markdown(f"**EchoSoul:** {reply}")
            st.caption(f"(Detected emotion: {emotion})")

            if save_memory:
                try:
                    enc_flag = 1 if cipher else 0
                    text_to_store = encrypt_text(user_input)
                    c.execute("INSERT INTO memories (title, content, encrypted, timestamp) VALUES (?, ?, ?, ?)",
                              (memory_title, text_to_store, enc_flag, datetime.now().isoformat()))
                    conn.commit()
                except Exception as e:
                    st.error(f"Save error: {e}")

            # Clear input safely
            st.session_state.chat_input = ""
            st.experimental_rerun()

# --- Memory Vault ---
elif mode == "Memory Vault":
    st.header("Memory Vault")
    c.execute("SELECT id, title, content, encrypted, timestamp FROM memories ORDER BY id DESC")
    rows = c.fetchall()
    for row in rows:
        with st.expander(f"{row[1]} ({row[4]})"):
            st.write(decrypt_text(row[2], row[3]))

# --- Life Timeline ---
elif mode == "Life Timeline":
    st.header("Life Timeline")
    c.execute("SELECT title, timestamp FROM memories ORDER BY timestamp ASC")
    rows = c.fetchall()
    for row in rows:
        st.markdown(f"**{row[0]}** — {row[1]}")

# --- Legacy Mode ---
elif mode == "Legacy Mode":
    st.header("Legacy Soul (Reflection Mode)")

    if "legacy_input" not in st.session_state:
        st.session_state.legacy_input = ""

    user_input = st.text_input("Ask your Legacy Soul", value=st.session_state.legacy_input, key="legacy_input_box")

    if st.button("Ask Legacy Soul"):
        if user_input.strip():
            context = "Past memories:\n"
            c.execute("SELECT content, encrypted FROM memories ORDER BY timestamp ASC LIMIT 5")
            rows = c.fetchall()
            for row in rows:
                context += decrypt_text(row[0], row[1]) + "\n"

            response = generate_echo_response(f"{context}\nUser: {user_input}\nLegacy Soul:", tone, "wise mentor")
            st.markdown(f"**Legacy Soul:** {response}")

            st.session_state.legacy_input = ""
            st.experimental_rerun()

# --- Soul Resonance ---
elif mode == "Soul Resonance":
    st.header("Soul Resonance Network")

    if "resonance_input" not in st.session_state:
        st.session_state.resonance_input = ""

    user_input = st.text_input("Send message to Resonance Network", value=st.session_state.resonance_input, key="resonance_input_box")

    if st.button("Send to Resonance"):
        if user_input.strip():
            response = generate_echo_response(user_input, tone, "collective consciousness")
            st.markdown(f"**Resonance Reply:** {response}")

            st.session_state.resonance_input = ""
            st.experimental_rerun()
