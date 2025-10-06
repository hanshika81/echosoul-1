import os
import datetime
import sqlite3
import streamlit as st
from cryptography.fernet import Fernet
from textblob import TextBlob
import openai

# --- Load environment keys ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# Encryption key setup
ECHOSOUL_KEY = os.getenv("ECHOSOUL_KEY")
if not ECHOSOUL_KEY:
    # Fallback auto-generate key for testing (not persistent across redeploys)
    ECHOSOUL_KEY = Fernet.generate_key().decode()
    print("⚠️ Warning: No ECHOSOUL_KEY set. Using a temporary key.")
fernet = Fernet(ECHOSOUL_KEY.encode())

# --- Database setup ---
DB_FILE = "echosoul.db"
conn = sqlite3.connect(DB_FILE, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    user_input TEXT,
    response TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# --- NLP + AI Response ---
def analyze_emotion(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.3:
        return "positive"
    elif polarity < -0.3:
        return "negative"
    return "neutral"

def generate_echo_response(user_input, tone="neutral", persona="default"):
    try:
        mood = analyze_emotion(user_input)
        system_prompt = f"You are EchoSoul, an adaptive AI with tone={tone}, persona={persona}, mood={mood}. Respond deeply and personally."
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"(Error generating response: {e})"

# --- Secure memory save/load ---
def save_memory(title, user_input, response):
    encrypted_input = fernet.encrypt(user_input.encode()).decode()
    encrypted_response = fernet.encrypt(response.encode()).decode()
    cursor.execute("INSERT INTO memories (title, user_input, response) VALUES (?, ?, ?)",
                   (title, encrypted_input, encrypted_response))
    conn.commit()

def load_memories():
    cursor.execute("SELECT id, title, user_input, response, timestamp FROM memories ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    results = []
    for row in rows:
        decrypted_input = fernet.decrypt(row[2].encode()).decode()
        decrypted_response = fernet.decrypt(row[3].encode()).decode()
        results.append((row[0], row[1], decrypted_input, decrypted_response, row[4]))
    return results

# --- Streamlit UI ---
st.set_page_config(page_title="EchoSoul AI", layout="wide")

st.sidebar.title("🌌 EchoSoul Settings")
mode = st.sidebar.radio("Choose a mode", ["Conversation", "Memory Vault", "Life Timeline", "Legacy Mode", "Soul Resonance"])

tone = st.sidebar.selectbox("Tone", ["neutral", "friendly", "wise", "playful"])
persona = st.sidebar.text_input("Persona style (e.g. mentor, friend, philosopher)", "default")

st.sidebar.markdown("---")
st.sidebar.info("🔐 Vault enabled" if ECHOSOUL_KEY else "❌ No vault key set")

# --- Conversation Tab ---
if mode == "Conversation":
    st.subheader("Talk to EchoSoul")

    user_input = st.text_area("Say something", key="chat_input")

    save_to_memory = st.checkbox("Save to memory (encrypted)")
    memory_title = st.text_input("Memory title", f"Memory {datetime.date.today()}")

    if st.button("Send to EchoSoul"):
        if user_input.strip():
            response = generate_echo_response(user_input, tone, persona)
            st.markdown(f"**EchoSoul:**\n\n{response}")

            if save_to_memory:
                try:
                    save_memory(memory_title, user_input, response)
                except Exception as e:
                    st.error(f"Save error: {e}")

            # Clear chatbox after sending
            st.session_state.chat_input = ""

# --- Memory Vault Tab ---
elif mode == "Memory Vault":
    st.subheader("🔐 Memory Vault")
    memories = load_memories()
    if not memories:
        st.info("No memories saved yet.")
    else:
        for mid, title, uin, resp, ts in memories:
            with st.expander(f"{title} ({ts})"):
                st.markdown(f"**You:** {uin}")
                st.markdown(f"**EchoSoul:** {resp}")

# --- Life Timeline Tab ---
elif mode == "Life Timeline":
    st.subheader("📜 Life Timeline")
    st.info("Chronological journey of your memories.")
    memories = load_memories()
    for mid, title, uin, resp, ts in sorted(memories, key=lambda x: x[4]):
        st.markdown(f"**{ts} – {title}**\n- You: {uin}\n- EchoSoul: {resp}")

# --- Legacy Mode Tab ---
elif mode == "Legacy Mode":
    st.subheader("🌙 Legacy Mode")
    st.info("A preserved form of your EchoSoul designed to carry your essence forward.")
    query = st.text_area("Ask your Legacy Soul", key="legacy_input")
    if st.button("Consult Legacy Soul"):
        if query.strip():
            memories = load_memories()
            context = " ".join([f"You: {m[2]} EchoSoul: {m[3]}" for m in memories[-5:]])
            response = generate_echo_response(context + "\n\n" + query, tone="wise", persona="legacy")
            st.markdown(f"**Legacy EchoSoul:** {response}")
            # Clear input
            st.session_state.legacy_input = ""

# --- Soul Resonance Tab ---
elif mode == "Soul Resonance":
    st.subheader("🌐 Soul Resonance Network")
    st.info("Connect EchoSouls to share wisdom and solve problems together.")
    multi_query = st.text_area("Enter your thought to resonate...", key="resonance_input")
    if st.button("Send to Resonance Network"):
        if multi_query.strip():
            response = generate_echo_response(multi_query, tone="collective", persona="network")
            st.markdown(f"**Resonance:** {response}")
            # Clear input
            st.session_state.resonance_input = ""
