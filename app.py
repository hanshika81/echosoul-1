import os
import streamlit as st
import sqlite3
from datetime import datetime
from cryptography.fernet import Fernet
from openai import OpenAI

# --- Setup ---
st.set_page_config(page_title="EchoSoul AI", layout="wide")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Database setup
conn = sqlite3.connect("echosoul.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memories
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT,
              content TEXT,
              encrypted INTEGER,
              timestamp TEXT)''')
conn.commit()

# Encryption setup
def get_cipher():
    key = os.getenv("ECHOSOUL_KEY")
    return Fernet(key.encode()) if key else None

cipher = get_cipher()

def encrypt_text(text):
    return cipher.encrypt(text.encode()).decode() if cipher else text

def decrypt_text(text):
    return cipher.decrypt(text.encode()).decode() if cipher else text

# --- Sidebar Controls ---
st.sidebar.title("⚡ EchoSoul Settings")

mode = st.sidebar.radio("Choose a mode", ["Conversation", "Memory Vault", "Life Timeline", "Legacy Mode", "Soul Resonance"])
tone = st.sidebar.selectbox("Tone", ["neutral", "playful", "empathetic", "serious"])
persona = st.sidebar.text_input("Persona style (e.g. mentor, friend, philosopher)", "default")

vault_status = "enabled" if cipher else "disabled"
st.sidebar.info(f"🔐 Vault {vault_status}")

# --- Core AI Functions ---
def detect_emotion(text):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "Classify the emotion in one word (e.g., happy, sad, angry, neutral)."},
                      {"role": "user", "content": text}]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "neutral"

def generate_echo_response(user_input, tone, persona):
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"You are EchoSoul, a {tone} companion with a {persona} style."},
                {"role": "user", "content": user_input}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"

# --- Conversation Mode ---
if mode == "Conversation":
    st.header("Talk to EchoSoul")

    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    user_input = st.text_input("Say something", value=st.session_state.chat_input, key="chat_input_box")

    save_memory = st.checkbox("Save to memory (encrypted)")
    memory_title = st.text_input("Memory title", f"Memory {datetime.now().date()}")

    if st.button("Send to EchoSoul"):
        if user_input.strip():
            emotion = detect_emotion(user_input)
            reply = generate_echo_response(user_input, tone, persona)

            st.markdown(f"**EchoSoul:** {reply}")
            st.caption(f"(Detected emotion: {emotion} 😐)")

            if save_memory:
                try:
                    enc_flag = 1 if cipher else 0
                    text_to_store = encrypt_text(user_input)
                    c.execute("INSERT INTO memories (title, content, encrypted, timestamp) VALUES (?, ?, ?, ?)",
                              (memory_title, text_to_store, enc_flag, datetime.now().isoformat()))
                    conn.commit()
                except Exception as e:
                    st.error(f"Save error: {e}")

            # ✅ Clear input safely
            st.session_state.chat_input = ""
            st.rerun()

# --- Memory Vault ---
elif mode == "Memory Vault":
    st.header("📜 Memory Vault")
    c.execute("SELECT id, title, content, encrypted, timestamp FROM memories ORDER BY timestamp DESC")
    rows = c.fetchall()

    for row in rows:
        mid, title, content, enc, ts = row
        if enc and cipher:
            try:
                content = decrypt_text(content)
            except Exception:
                content = "[Decryption failed]"
        elif enc and not cipher:
            content = "[Encrypted]"

        with st.expander(f"{title} ({ts})"):
            st.write(content)

# --- Life Timeline ---
elif mode == "Life Timeline":
    st.header("📅 Life Timeline")
    c.execute("SELECT title, timestamp FROM memories ORDER BY timestamp ASC")
    rows = c.fetchall()
    for title, ts in rows:
        st.markdown(f"- **{ts}** — {title}")

# --- Legacy Mode ---
elif mode == "Legacy Mode":
    st.header("🌌 Legacy Mode (Reflective Echoes)")

    if "legacy_input" not in st.session_state:
        st.session_state.legacy_input = ""

    user_input = st.text_input("Speak to your legacy self", value=st.session_state.legacy_input, key="legacy_input_box")

    if st.button("Send to Legacy Echo"):
        if user_input.strip():
            reply = generate_echo_response(user_input, "philosophical", "wise elder")
            st.markdown(f"**Legacy Echo:** {reply}")

            st.session_state.legacy_input = ""
            st.rerun()

# --- Soul Resonance ---
elif mode == "Soul Resonance":
    st.header("💫 Soul Resonance (Deep Echo Connection)")

    if "resonance_input" not in st.session_state:
        st.session_state.resonance_input = ""

    user_input = st.text_input("Share your deepest thoughts", value=st.session_state.resonance_input, key="resonance_input_box")

    if st.button("Send to Resonance Echo"):
        if user_input.strip():
            reply = generate_echo_response(user_input, "empathetic", "soulful guide")
            st.markdown(f"**Resonance Echo:** {reply}")

            st.session_state.resonance_input = ""
            st.rerun()
