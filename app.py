import os
import json
import streamlit as st
import datetime
from cryptography.fernet import Fernet
from textblob import TextBlob
import openai
from openai.error import RateLimitError, OpenAIError

# ----------------------------
# Setup
# ----------------------------
st.set_page_config(
    page_title="EchoSoul",
    page_icon="✨",
    layout="wide"
)

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Encryption Key
VAULT_KEY = os.getenv("ECHOSOUL_KEY")
if VAULT_KEY:
    try:
        fernet = Fernet(VAULT_KEY)
        vault_enabled = True
    except Exception:
        fernet = None
        vault_enabled = False
else:
    fernet = None
    vault_enabled = False

# ----------------------------
# Utility Functions
# ----------------------------
def encrypt_memory(data: str) -> str:
    if not vault_enabled:
        return data
    return fernet.encrypt(data.encode()).decode()

def decrypt_memory(token: str) -> str:
    if not vault_enabled:
        return token
    return fernet.decrypt(token.encode()).decode()

def analyze_emotion(text: str) -> str:
    tb = TextBlob(text)
    polarity = tb.sentiment.polarity
    if polarity > 0.5:
        return "very positive"
    elif polarity > 0:
        return "positive"
    elif polarity == 0:
        return "neutral"
    elif polarity < -0.5:
        return "very negative"
    else:
        return "negative"

def call_openai(prompt: str, system_prompt: str = "You are EchoSoul, an adaptive AI companion.") -> str:
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return resp.choices[0].message["content"].strip()
    except RateLimitError:
        return "⚠️ OpenAI API rate limit reached. Please try again later."
    except OpenAIError as e:
        return f"⚠️ OpenAI API error: {str(e)}"
    except Exception as e:
        return f"⚠️ Unexpected error: {str(e)}"

# ----------------------------
# State Initialization
# ----------------------------
if "memories" not in st.session_state:
    st.session_state.memories = []
if "timeline" not in st.session_state:
    st.session_state.timeline = []
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "profile" not in st.session_state:
    st.session_state.profile = {
        "name": "",
        "tone": "neutral",
        "persona": "A supportive companion."
    }

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("EchoSoul Controls")
st.sidebar.text_input("Your name", key="profile_name", value=st.session_state.profile["name"])
st.sidebar.selectbox("Personality tone", ["neutral", "playful", "serious", "empathetic"], key="profile_tone", index=["neutral", "playful", "serious", "empathetic"].index(st.session_state.profile["tone"]))
st.sidebar.text_area("Persona description (how EchoSoul sounds)", key="profile_persona", value=st.session_state.profile["persona"])
if st.sidebar.button("Save profile"):
    st.session_state.profile["name"] = st.session_state.profile_name
    st.session_state.profile["tone"] = st.session_state.profile_tone
    st.session_state.profile["persona"] = st.session_state.profile_persona
    st.sidebar.success("Profile updated!")

st.sidebar.markdown("### Encryption / Vault")
if vault_enabled:
    st.sidebar.success("Vault enabled — encrypted memories will be saved.")
else:
    st.sidebar.warning("Vault disabled — memories will not be encrypted.")

# ----------------------------
# Main Tabs
# ----------------------------
tabs = st.tabs([
    "Conversation",
    "Memory Vault",
    "Life Timeline",
    "Time-Shifted Self",
    "Legacy Mode"
])

# ----------------------------
# Conversation Tab
# ----------------------------
with tabs[0]:
    st.header("Talk to EchoSoul")
    user_input = st.text_area("Say something", key="chat_input")
    save_memory = st.checkbox("Save to memory (encrypted)")
    memory_title = st.text_input("Memory title", value=f"Memory {datetime.date.today()}")

    if st.button("Send to EchoSoul"):
        if user_input.strip():
            emotion = analyze_emotion(user_input)
            system_prompt = f"You are EchoSoul, an adaptive AI companion. Tone: {st.session_state.profile['tone']}. Persona: {st.session_state.profile['persona']}."
            reply = call_openai(f"User: {user_input}\nEmotion detected: {emotion}\nRespond helpfully.")
            st.session_state.conversation.append(("user", user_input))
            st.session_state.conversation.append(("ai", reply))

            if save_memory:
                encrypted = encrypt_memory(user_input + " | " + reply)
                st.session_state.memories.append({
                    "title": memory_title,
                    "content": encrypted,
                    "date": str(datetime.date.today())
                })

    st.subheader("Conversation History")
    for speaker, text in st.session_state.conversation:
        if speaker == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**EchoSoul:** {text}")

# ----------------------------
# Memory Vault Tab
# ----------------------------
with tabs[1]:
    st.header("Your Memory Vault")
    if not st.session_state.memories:
        st.info("No memories saved yet.")
    else:
        for mem in st.session_state.memories:
            with st.expander(mem["title"] + " (" + mem["date"] + ")"):
                if vault_enabled:
                    st.write(decrypt_memory(mem["content"]))
                else:
                    st.write(mem["content"])

# ----------------------------
# Life Timeline Tab
# ----------------------------
with tabs[2]:
    st.header("Your Life Timeline")
    timeline_entry = st.text_input("Add a life event")
    if st.button("Save event"):
        if timeline_entry.strip():
            st.session_state.timeline.append({
                "event": timeline_entry,
                "date": str(datetime.date.today())
            })
    if not st.session_state.timeline:
        st.info("No timeline events yet.")
    else:
        for event in st.session_state.timeline:
            st.markdown(f"📌 **{event['date']}** — {event['event']}")

# ----------------------------
# Time-Shifted Self Tab
# ----------------------------
with tabs[3]:
    st.header("Talk to Your Past or Future Self")
    target = st.radio("Choose", ["Past Self", "Future Self"])
    question = st.text_area("What do you want to ask?")
    if st.button("Ask Time-Shifted Self"):
        if question.strip():
            direction = "Imagine you are the user's past self." if target == "Past Self" else "Imagine you are the user's future self."
            answer = call_openai(f"{direction} The user asks: {question}")
            st.markdown(f"**{target} says:** {answer}")

# ----------------------------
# Legacy Mode Tab
# ----------------------------
with tabs[4]:
    st.header("Legacy Mode")
    st.markdown("Preserve your wisdom for future generations.")
    legacy_prompt = st.text_area("What message would you like to leave behind?")
    if st.button("Save Legacy Message"):
        if legacy_prompt.strip():
            st.session_state.memories.append({
                "title": f"Legacy - {datetime.date.today()}",
                "content": encrypt_memory(legacy_prompt),
                "date": str(datetime.date.today())
            })
            st.success("Legacy message saved to vault!")
    if st.session_state.memories:
        st.subheader("Legacy Messages")
        for mem in st.session_state.memories:
            if mem["title"].startswith("Legacy"):
                with st.expander(mem["title"]):
                    if vault_enabled:
                        st.write(decrypt_memory(mem["content"]))
                    else:
                        st.write(mem["content"])
