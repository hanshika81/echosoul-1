import os
import sqlite3
from datetime import datetime
from cryptography.fernet import Fernet
import streamlit as st
from textblob import TextBlob
from openai import OpenAI

# -----------------------
# Setup
# -----------------------
st.set_page_config(page_title="EchoSoul AI", layout="wide")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ECHOSOUL_KEY = os.getenv("ECHOSOUL_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# Encryption
cipher = Fernet(ECHOSOUL_KEY.encode()) if ECHOSOUL_KEY else None

# SQLite setup
conn = sqlite3.connect("echosoul_memories.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS memories
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              title TEXT,
              content TEXT,
              encrypted INTEGER,
              timestamp TEXT)''')
conn.commit()

# -----------------------
# Helpers
# -----------------------
def encrypt_text(text):
    return cipher.encrypt(text.encode()).decode() if cipher else text

def decrypt_text(text, encrypted):
    return cipher.decrypt(text.encode()).decode() if cipher and encrypted else text

def detect_emotion(text):
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0.3: return "positive 😊"
    elif sentiment < -0.3: return "negative 😔"
    return "neutral 😐"

def generate_echo_response(user_input, tone, persona):
    """Use OpenAI NLP for generating companion response"""
    prompt = f"""
    You are EchoSoul, a personal adaptive companion.
    Tone: {tone}
    Persona style: {persona}

    The user said: "{user_input}"
    Respond in a supportive, conversational way.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(Error: {e})"

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("⚡ EchoSoul Settings")
mode = st.sidebar.radio("Choose a mode", 
    ["Conversation", "Memory Vault", "Life Timeline", "Legacy Mode", "Soul Resonance"])

tone = st.sidebar.selectbox("Tone", ["neutral", "warm", "playful", "serious", "philosophical"])
persona = st.sidebar.text_input("Persona style (e.g. mentor, friend, philosopher)", value="default")

if ECHOSOUL_KEY:
    st.sidebar.success("🔐 Vault enabled")
else:
    st.sidebar.warning("🔑 ECHOSOUL_KEY not set. Vault disabled.")

# -----------------------
# Main Modes
# -----------------------

# 1. Conversation Mode
if mode == "Conversation":
    st.title("EchoSoul — personal adaptive companion (demo)")
    st.subheader("Talk to EchoSoul")

    # Safe chat input
    user_input = st.text_input("Say something", key="chat_input")
    save_memory = st.checkbox("Save to memory (encrypted)")
    memory_title = st.text_input("Memory title", value=f"Memory {datetime.now().date()}")

    if st.button("Send to EchoSoul"):
        if user_input:
            emotion = detect_emotion(user_input)
            reply = generate_echo_response(user_input, tone, persona)

            st.markdown(f"**EchoSoul:** {reply}")
            st.caption(f"(Detected emotion: {emotion})")

            # Save memory
            if save_memory:
                enc_flag = 1 if cipher else 0
                text_to_store = encrypt_text(user_input)
                c.execute("INSERT INTO memories (title, content, encrypted, timestamp) VALUES (?, ?, ?, ?)",
                          (memory_title, text_to_store, enc_flag, datetime.now().isoformat()))
                conn.commit()

            # Clear input safely
            st.session_state.chat_input = ""

# 2. Memory Vault
elif mode == "Memory Vault":
    st.title("🔒 Memory Vault")
    st.write("Your stored (and encrypted) memories:")

    rows = c.execute("SELECT id, title, content, encrypted, timestamp FROM memories").fetchall()
    for row in rows:
        st.markdown(f"**{row[1]}** ({row[4]})")
        st.write(decrypt_text(row[2], row[3]))

# 3. Life Timeline
elif mode == "Life Timeline":
    st.title("📜 Life Timeline")
    st.write("Chronological view of your memories:")

    rows = c.execute("SELECT title, content, encrypted, timestamp FROM memories ORDER BY timestamp DESC").fetchall()
    for row in rows:
        st.markdown(f"**{row[0]}** ({row[3]})")
        st.write(decrypt_text(row[1], row[2]))

# 4. Legacy Mode
elif mode == "Legacy Mode":
    st.title("🌌 Legacy Mode")
    st.write("EchoSoul answers with wisdom from all past memories.")

    legacy_input = st.text_input("Ask the Legacy Soul", key="legacy_input")

    if st.button("Ask Legacy Soul"):
        if legacy_input:
            memories = c.execute("SELECT content, encrypted FROM memories").fetchall()
            memory_text = "\n".join([decrypt_text(m[0], m[1]) for m in memories])

            prompt = f"""
            You are the Legacy Soul, a wise and reflective persona.
            Use the following past memories to answer thoughtfully:
            {memory_text}

            Question: {legacy_input}
            """
            reply = generate_echo_response(prompt, "philosophical", "legacy mentor")
            st.markdown(f"**Legacy Soul:** {reply}")

            # Clear input
            st.session_state.legacy_input = ""

# 5. Soul Resonance
elif mode == "Soul Resonance":
    st.title("🌐 Soul Resonance Network")
    st.write("Connect with collective EchoSouls for expanded insight.")

    resonance_input = st.text_input("Say something to Resonance Network", key="resonance_input")

    if st.button("Send to Resonance"):
        if resonance_input:
            reply = generate_echo_response(resonance_input, "collaborative", "networked souls")
            st.markdown(f"**Resonance Response:** {reply}")

            # Clear input
            st.session_state.resonance_input = ""
