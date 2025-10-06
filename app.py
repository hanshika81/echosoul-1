import os
import json
import sqlite3
import streamlit as st
from datetime import datetime
from cryptography.fernet import Fernet, InvalidToken
from textblob import TextBlob

# Optional: for microphone capture in the browser (simulated live call)
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
    STREAMLIT_WEBRTC_AVAILABLE = True
except Exception:
    STREAMLIT_WEBRTC_AVAILABLE = False

# ---- Configuration & secrets ----
DB_PATH = os.environ.get("ECHOSOUL_DB", "echosoul.db")
# Encryption key - set as environment variable ECHOSOUL_KEY (generate with: Fernet.generate_key().decode())
FERNET_KEY = os.environ.get("ECHOSOUL_KEY")

# Optional OpenAI integration (placeholder)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# ---- Utilities ----
def get_fernet():
    if not FERNET_KEY:
        return None
    return Fernet(FERNET_KEY.encode() if isinstance(FERNET_KEY, str) else FERNET_KEY)

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

def encrypt_text(plain: str) -> bytes:
    f = get_fernet()
    if not f:
        raise RuntimeError("Encryption key not configured (ECHOSOUL_KEY).")
    return f.encrypt(plain.encode('utf-8'))

def decrypt_bytes(blob: bytes) -> str:
    f = get_fernet()
    if not f:
        raise RuntimeError("Encryption key not configured (ECHOSOUL_KEY).")
    try:
        return f.decrypt(blob).decode('utf-8')
    except InvalidToken:
        return "<DECRYPTION FAILED - INVALID KEY>"

def save_memory(title: str, content: str, encrypt: bool=False, metadata: dict=None):
    cur = conn.cursor()
    created_at = datetime.utcnow().isoformat()
    metadata = json.dumps(metadata or {})
    if encrypt:
        blob = encrypt_text(content)
        cur.execute("INSERT INTO memories (created_at, title, content, encrypted, metadata) VALUES (?, ?, ?, 1, ?)",
                    (created_at, title, blob, metadata))
    else:
        cur.execute("INSERT INTO memories (created_at, title, content, encrypted, metadata) VALUES (?, ?, ?, 0, ?)",
                    (created_at, title, content, 0, metadata))
    conn.commit()

def list_memories():
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, title, encrypted, metadata FROM memories ORDER BY created_at DESC")
    rows = cur.fetchall()
    results = []
    for r in rows:
        results.append({
            "id": r[0],
            "created_at": r[1],
            "title": r[2],
            "encrypted": bool(r[3]),
            "metadata": json.loads(r[4] or "{}")
        })
    return results

def get_memory(mid: int):
    cur = conn.cursor()
    cur.execute("SELECT id, created_at, title, content, encrypted, metadata FROM memories WHERE id = ?", (mid,))
    r = cur.fetchone()
    if not r:
        return None
    content = r[3]
    if r[4]:  # encrypted
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

def save_profile(profile: dict):
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO profile (id, data) VALUES (1, ?)", (json.dumps(profile),))
    conn.commit()

def load_profile():
    cur = conn.cursor()
    cur.execute("SELECT data FROM profile WHERE id = 1")
    r = cur.fetchone()
    if not r: return {}
    return json.loads(r[0])

# ---- Simple emotion detector (text) ----
def detect_emotion_text(text: str) -> str:
    # Very simple heuristic using TextBlob sentiment for demo purposes.
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.4:
        return "happy/positive"
    elif polarity > 0.05:
        return "calm/content"
    elif polarity < -0.4:
        return "angry/sad"
    elif polarity < -0.05:
        return "upset/negative"
    else:
        return "neutral/ambivalent"

# ---- Simple adaptive response generator (placeholder) ----
def generate_echo_response(user_text: str, profile: dict, mode: str="reflect"):
    # Placeholder. Replace with call to your LLM or OpenAI.
    # Uses profile['tone'] or profile['persona_description'] to adapt style.
    tone = profile.get("tone", "warm")
    persona = profile.get("persona_description", "A supportive companion.")
    emotion = detect_emotion_text(user_text)
    response = f"[{tone} voice | detected: {emotion}] I hear you. You said: \"{user_text}\".\n\n(Adaptive persona: {persona})"
    return response

# ---- Streamlit UI ----
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("EchoSoul — personal adaptive companion (demo)")

# Sidebar: profile & controls
with st.sidebar:
    st.header("EchoSoul Controls")
    profile = load_profile()
    name = st.text_input("Your name", value=profile.get("name",""))
    tone = st.selectbox("Personality tone", options=["warm","scientific","playful","stoic","empathetic"], index=0)
    persona = st.text_area("Persona description (how EchoSoul sounds)", value=profile.get("persona_description","A supportive companion."))
    if st.button("Save profile"):
        profile = {"name": name, "tone": tone, "persona_description": persona}
        save_profile(profile)
        st.success("Saved profile")

    st.markdown("---")
    st.markdown("**Encryption**")
    if FERNET_KEY:
        st.success("Encryption key configured")
        st.write("Encrypted memory is available.")
    else:
        st.warning("ECHOSOUL_KEY not set. Encrypted vault disabled.")
    st.markdown("---")
    st.write("Deploy notes:")
    st.caption("Set `ECHOSOUL_KEY` (Fernet key) and optional `OPENAI_API_KEY` in Render environment.")

# Main area: tabs
tab = st.tabs(["Conversation", "Memory Vault", "Life Timeline", "Time-Shifted Self", "Live Call (Sim)"])

# Conversation tab
with tab[0]:
    st.header("Talk to EchoSoul")
    profile = load_profile()
    user_input = st.text_area("Say something to your EchoSoul", height=120)
    col1, col2 = st.columns([1,1])
    with col1:
        encrypt_mem = st.checkbox("Save this to memory (encrypted)", value=False)
        remember_title = st.text_input("Memory title", value=f"Memory {datetime.utcnow().date()}")

    with col2:
        st.write("Quick tools")
        st.button("Detect emotion", key="emo_button")
        st.write("Detected emotion (live):")
        if user_input.strip():
            st.info(detect_emotion_text(user_input))
        else:
            st.write("—")

    if st.button("Send to EchoSoul"):
        if not user_input.strip():
            st.warning("Write something first.")
        else:
            # generate a response (placeholder)
            response = generate_echo_response(user_input, profile)
            st.markdown("**EchoSoul:**")
            st.write(response)

            # save memory optionally
            if st.session_state.get("remember_now", True) or encrypt_mem:
                try:
                    save_memory(remember_title or "Quick memory", user_input, encrypt=encrypt_mem,
                                metadata={"source": "conversation", "tone": profile.get("tone")})
                    st.success("Saved to memory" + (" (encrypted)" if encrypt_mem else ""))
                except Exception as e:
                    st.error(f"Failed to save memory: {e}")

# Memory Vault
with tab[1]:
    st.header("Memory Vault")
    mems = list_memories()
    left, right = st.columns([1,2])
    with left:
        st.subheader("Your memories")
        for m in mems:
            label = f"{m['id']}: {m['title']} — {m['created_at'][:19]}"
            if st.button(label, key=f"open_{m['id']}"):
                st.session_state["open_memory"] = m['id']
    with right:
        mid = st.session_state.get("open_memory")
        if mid:
            mem = get_memory(mid)
            if mem:
                st.subheader(mem["title"])
                st.write("Created at:", mem["created_at"])
                if mem["encrypted"]:
                    if not FERNET_KEY:
                        st.error("Memory is encrypted but no key is configured (ECHOSOUL_KEY).")
                    else:
                        st.success("Encrypted memory (decrypted below):")
                st.markdown(mem["content"])
                st.markdown("Metadata:")
                st.json(mem["metadata"])
                if st.button("Reflect on this memory"):
                    resp = generate_echo_response(mem["content"], profile)
                    st.write(resp)
        else:
            st.info("Select a memory from the list on the left or add a new one below.")
    st.markdown("---")
    st.subheader("Add new memory")
    new_title = st.text_input("Title for memory", value=f"Moment {datetime.utcnow().isoformat()}")
    new_content = st.text_area("Memory content")
    new_encrypt = st.checkbox("Encrypt memory", value=False, key="new_enc")
    if st.button("Save memory"):
        if not new_content.strip():
            st.warning("Write something to save.")
        else:
            try:
                save_memory(new_title, new_content, encrypt=new_encrypt)
                st.success("Memory saved.")
            except Exception as e:
                st.error(f"Failed: {e}")

# Life Timeline
with tab[2]:
    st.header("Life Timeline")
    mems = list_memories()
    st.write("Chronological view of saved memories (newest first).")
    for m in mems:
        st.markdown(f"**{m['title']}** — {m['created_at'][:19]}")
        if st.button(f"Open {m['id']}", key=f"tl_{m['id']}"):
            st.session_state["open_memory"] = m['id']

# Time-Shifted Self
with tab[3]:
    st.header("Time-Shifted Self (simulator)")
    st.write("Simulate talking to your past or future EchoSoul persona.")
    mode = st.radio("Mode", ["past", "future", "reflection"])
    years = st.slider("Years shift (positive integer)", min_value=1, max_value=50, value=5)
    prompt = st.text_area("What would you ask your time-shifted self?")
    if st.button("Talk to time-shifted self"):
        persona = load_profile().get("persona_description", "")
        # A basic simulation: vary persona slightly with mode and years
        simulated_persona = f"{persona} (as your {mode} self, {years} years {'ago' if mode=='past' else 'ahead'})"
        response = generate_echo_response(prompt or "Hello", {"tone": profile.get("tone"), "persona_description": simulated_persona})
        st.write(response)

# Live Call (Sim)
with tab[4]:
    st.header("Live Call (Simulated)")
    st.write("This demo uses browser mic capture (if available) to show how live call emotion detection might work.")
    if not STREAMLIT_WEBRTC_AVAILABLE:
        st.warning("streamlit-webrtc is not installed or not available in this environment. Install `streamlit-webrtc` and add ffmpeg in apt.txt for full functionality.")
        st.info("On Render, you'll need ffmpeg system package (see apt.txt) and `streamlit-webrtc` in requirements.")
    else:
        st.write("Click Start to capture audio and run a simple emotion heuristic.")
        class SimpleAudioProcessor(AudioProcessorBase):
            def recv(self, frame):
                # placeholder: we would run VAD/emotion detection here
                return frame

        webrtc_streamer(key="echosoul-call", audio_processor_factory=SimpleAudioProcessor)
        st.markdown("**Captured audio** will be processed (placeholder).")
        st.caption("Replace with your chosen provider (Twilio/Agora) for real PSTN/VoIP calls.")

# Footer
st.markdown("---")
st.write("EchoSoul demo — scaffold for the features you described. Replace `generate_echo_response` with calls to your LLM provider and expand voice/emotion pipelines for production.")
