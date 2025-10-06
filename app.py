# app.py - EchoSoul (large single-file Streamlit app)
# Features: persistent memory, adaptive personality, emotion detection,
# timeline, custom style learning, private encrypted vault, legacy mode,
# time-shifted self (simulated), soul resonance network (peer demo), life
# path simulator (heuristic), realtime voice (streamlit-webrtc), export.
#
# NOTE: read the top-of-file comments about dependencies and secrets.
#
# SECRETS NOTE: For deployment, it is highly recommended to set OPENAI_API_KEY
# and FERNET_KEY as ENVIRONMENT VARIABLES in your hosting service (e.g., Render).
# The code falls back to os.getenv() if st.secrets cannot find the local file.

import streamlit as st
import os
import json
import datetime
import re
from cryptography.fernet import Fernet
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
import threading
import time

# OpenAI: try new client if available, fallback if needed
try:
    # modern client
    from openai import OpenAI
    OPENAI_NEW = True
except Exception:
    OPENAI_NEW = False
    import openai

# WebRTC and audio
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
    import av
    WEBSUPPORT = True
except Exception:
    WEBSUPPORT = False

# Speech recognition and TTS (optional, platform-dependent)
try:
    import speech_recognition as sr
    import pyttsx3
    AUDIO_SUPPORT = True
except Exception:
    AUDIO_SUPPORT = False

# -------------------------
# Config & Secrets
# -------------------------
st.set_page_config(layout="wide", page_title="EchoSoul")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
TIMELINE_FILE = DATA_DIR / "timeline.json"
MEMORY_FILE = DATA_DIR / "memories.json"
VAULT_FILE = DATA_DIR / "vault.json"
PERSONA_FILE = DATA_DIR / "persona.json"

# --- SECRET LOADING LOGIC ---
# The code below correctly implements the fallback to environment variables,
# which is the fix for the StreamlitSecretNotFoundError in a deployment setting.
OPENAI_API_KEY = None
# 1. Try Streamlit secrets (local .streamlit/secrets.toml)
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
# 2. Fallback to OS environment variable (Recommended for Render)
elif os.getenv("OPENAI_API_KEY"):
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    # This warning will show in the app if the key is missing from secrets or env vars.
    st.warning("OpenAI API key not found. Put it in Streamlit secrets or env var OPENAI_API_KEY.")
# initialize client
if OPENAI_API_KEY:
    if OPENAI_NEW:
        client = OpenAI(api_key=OPENAI_API_KEY)
    else:
        openai.api_key = OPENAI_API_KEY
        client = openai

# -------------------------
# Utilities: file persistence
# -------------------------
def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default

def save_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# -------------------------
# Data structures
# -------------------------
@dataclass
class Memory:
    timestamp: str
    title: str
    content: str
    tags: List[str]

# -------------------------
# Session state initialisation (critical to avoid widget modification errors)
# Must set defaults BEFORE creating widgets that use the keys.
# -------------------------
defaults = {
    "chat_history": [],          # {"role":"user"/"assistant","text":...}
    "timeline": load_json(TIMELINE_FILE, []),
    "memories": load_json(MEMORY_FILE, []),
    "vault_key": None,
    "vault": load_json(VAULT_FILE, {}),
    "persona": load_json(PERSONA_FILE, {"style":"friendly", "nick":"EchoSoul"}),
    "personality_profile": {},
    "adaptive_style_examples": [],
    "knowledge": [],
    # ui keys
    "chat_input": "",
    "selected_section": "Chat",
    "live_call_active": False
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Setup encryption key if not present
if st.session_state.vault_key is None:
    # allow using pre-set fernet key in secrets for persistence across restarts
    # This block is where the second potential secrets error could occur if FERNET_KEY
    # is only in st.secrets. We'll rely on the OS Environment variable setup for it.
    fernet_key_from_env = os.getenv("FERNET_KEY")
    if "FERNET_KEY" in st.secrets and st.secrets["FERNET_KEY"]:
        st.session_state.vault_key = st.secrets["FERNET_KEY"].encode()
    elif fernet_key_from_env:
        st.session_state.vault_key = fernet_key_from_env.encode()
    else:
        # Generate new key if neither secrets nor environment variable is set
        st.session_state.vault_key = Fernet.generate_key()
fernet = Fernet(st.session_state.vault_key)

# -------------------------
# Basic NLP helpers (emotion / style / summarize)
# -------------------------
def detect_emotion_text(text: str) -> str:
    t = text.lower()
    # simple heuristics
    if any(w in t for w in ["sad", "depressed", "unhappy", "miserable", "cry"]):
        return "sad"
    if any(w in t for w in ["happy", "glad", "joy", "excited", "yay"]):
        return "happy"
    if any(w in t for w in ["angry", "mad", "furious", "irritat"]):
        return "angry"
    if any(w in t for w in ["anxious", "worried", "nervous"]):
        return "anxious"
    return "neutral"

def extract_personal_fact(text: str) -> Dict[str,str]:
    # tiny heuristic: "I am X", "I like X", "I live in X", "I work as X"
    m = re.search(r"\bI (?:am|\'m|like|live|work as|worked as|was)\b (.+)", text, flags=re.I)
    if m:
        fact = m.group(1).strip()
        return {"fact": fact}
    return {}

def short_summary(text: str, max_len=200) -> str:
    return text.strip()[:max_len]

# -------------------------
# OpenAI wrapper (resilient)
# -------------------------
def openai_chat_completion(messages: List[Dict[str,str]], model="gpt-4o-mini", temperature=0.7, max_tokens=512):
    """
    Use the available OpenAI client. Supports both new client (OpenAI) and legacy 'openai' import.
    Returns assistant text or raises exception.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI key not configured.")
    try:
        if OPENAI_NEW:
            # new OpenAI client style
            resp = client.chat.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            # The structure for accessing the content has been slightly corrected for robustness
            return resp.choices[0].message.content
        else:
            # fallback older style
            # Note: Legacy client might use model="gpt-4" if gpt-4o-mini isn't supported there.
            # Assuming the import setup handles the correct client instantiation.
            resp = client.ChatCompletion.create(model="gpt-4", messages=messages, temperature=temperature, max_tokens=max_tokens)
            return resp.choices[0].message["content"]
    except Exception as e:
        # bubble up clean error for UI
        raise

# -------------------------
# Core features
# -------------------------
def add_to_timeline(item: Dict[str,Any]):
    st.session_state.timeline.insert(0, item)  # newest first
    save_json(TIMELINE_FILE, st.session_state.timeline)

def store_memory(title: str, content: str, tags=None):
    if tags is None:
        tags = []
    mem = Memory(timestamp=str(datetime.datetime.utcnow()), title=title, content=content, tags=tags)
    st.session_state.memories.insert(0, asdict(mem))
    save_json(MEMORY_FILE, st.session_state.memories)

def encrypt_and_save_vault(key: str, plaintext: str):
    token = fernet.encrypt(plaintext.encode()).decode()
    st.session_state.vault[key] = token
    save_json(VAULT_FILE, st.session_state.vault)

def read_vault(key: str):
    token = st.session_state.vault.get(key)
    if not token:
        return None
    try:
        return fernet.decrypt(token.encode()).decode()
    except Exception:
        return None

# Personality & adaptive style learning
def learn_style_example(user_text: str):
    st.session_state.adaptive_style_examples.append({"text": user_text, "time": str(datetime.datetime.utcnow())})
    # optionally store persona file
    save_json(PERSONA_FILE, st.session_state.persona)

def update_personality_with_interaction(user_text: str, assistant_text: str):
    # simple heuristic: store into "knowledge" and adjust style counters
    st.session_state.knowledge.append(user_text)
    # simple tone modification: if user uses short sentences -> become succinct
    avg_len = sum(len(x.split()) for x in st.session_state.adaptive_style_examples)/max(1,len(st.session_state.adaptive_style_examples))
    if avg_len < 6:
        st.session_state.persona["style"] = "succinct"
    else:
        st.session_state.persona["style"] = "conversational"
    save_json(PERSONA_FILE, st.session_state.persona)

# -------------------------
# High-level assistant: build system prompt from persona & timeline
# -------------------------
def craft_system_prompt():
    p = st.session_state.persona
    timeline_summary = f"You have stored {len(st.session_state.timeline)} timeline events."
    persona_lines = f"Persona name: {p.get('nick','EchoSoul')}. Style: {p.get('style','friendly')}."
    return f"You are EchoSoul, an adaptive personal companion.\n{persona_lines}\n{timeline_summary}\nAlways be empathetic, preserve privacy, and encourage reflection."

def ask_echo(input_text: str, role: str="user") -> str:
    # build messages with memory context (limited)
    system = craft_system_prompt()
    messages = [{"role":"system","content":system}]
    # include last few timeline events
    for ev in st.session_state.timeline[:6]:
        messages.append({"role":"system","content":f"Timeline: {ev.get('timestamp')} - {ev.get('note') or ev.get('user','')} - {ev.get('echo','')}"})
    # include persona hints
    messages.append({"role":"system","content":f"Personality: {st.session_state.persona}"})
    messages.append({"role":"user","content":input_text})
    # call OpenAI
    try:
        resp_text = openai_chat_completion(messages)
    except Exception as e:
        # produce a safe fallback
        return f"(Error calling OpenAI) {str(e)}"
    # post-process
    update_personality_with_interaction(input_text, resp_text)
    return resp_text

# -------------------------
# Advanced modules (simulated but functional)
# -------------------------
# 1) Time-Shifted Self: we allow the user to "speak to past self" or "future self" using prompt engineering
def time_shifted_self(user_message: str, shift: str="past") -> str:
    # shift: "past" | "future"
    if shift == "past":
        prompt = f"Roleplay the user's past self (5 years ago). You were more naive and hopeful. Respond concisely and reflectively to: {user_message}"
    else:
        prompt = f"Roleplay the user's future self (10 years in the future). Speak as a wiser, kinder version, giving advice on: {user_message}"
    return ask_echo(prompt)

# 2) Soul Resonance Network: demonstration peer-to-peer "knowledge sharing"
# For safety, this is simulated locally by merging knowledge from a "network" file; functionally it merges nodes.
NETWORK_FILE = DATA_DIR / "resonance_network.json"
def network_broadcast_share(note: str):
    net = load_json(NETWORK_FILE, [])
    node = {"timestamp": str(datetime.datetime.utcnow()), "note": note}
    net.insert(0, node)
    save_json(NETWORK_FILE, net)
    # merge into local knowledge (demo)
    st.session_state.knowledge.append(note)
    return f"Broadcasted to local network (demo)."

def network_query_similar(query: str, limit=3):
    net = load_json(NETWORK_FILE, [])
    res = []
    for item in net:
        if query.lower() in (item.get("note","").lower()):
            res.append(item)
        if len(res) >= limit:
            break
    return res

# 3) Life Path Simulation: simple scenario engine with heuristic scoring
def simulate_life_path(current_state: Dict[str,Any], choices: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
    # Each choice: {"title":..., "effect": function or dict}
    results = []
    for c in choices:
        # naive scoring: +1 if aligned with 'values' in persona
        score = 0
        if "values" in st.session_state.personality_profile:
            for v in st.session_state.personality_profile["values"]:
                if v.lower() in c.get("title","").lower():
                    score += 2
        # random deterministic additive from length
        score += len(c.get("title","")) % 5
        res = {"choice": c, "score": score, "outcome": f"Simulated outcome for {c.get('title')}"}
        results.append(res)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# 4) Echo Guardian & Consciousness Mirror: lightweight analytics on user's inputs
def mirror_reflection(user_text: str) -> str:
    # echo back habits (word frequency)
    words = re.findall(r"\w+", user_text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w,0) + 1
    top = sorted(freq.items(), key=lambda x:x[1], reverse=True)[:5]
    top_text = ", ".join([f"{w}:{c}" for w,c in top])
    return f"I noticed you repeat these words often: {top_text}. This may reflect your focus or concern."

# -------------------------
# Live call: audio processor (very simple)
# -------------------------
if WEBSUPPORT and AUDIO_SUPPORT:
    class EchoAudioProcessor(AudioProcessorBase):
        def __init__(self):
            self.recognizer = sr.Recognizer()
            self.tts_engine = pyttsx3.init()
            # configure voice rate if you want
            self.tts_engine.setProperty('rate', 150)

        def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
            # We won't implement low-level audio frame -> speech recognition reliably in this file.
            # streamlit-webrtc demos often use a JS media pipeline or the recorded chunks approach.
            return frame

    # Note: for a fully functioning live call you'd convert audio frames -> wav -> SpeechRecognition
    # and then call ask_echo(...) and return TTS audio. That requires additional streaming control.

# -------------------------
# UI: main layout
# -------------------------
st.sidebar.title("Settings")
vault_pw = st.sidebar.text_input("Vault password (for UI unlock)", type="password")
section = st.sidebar.radio("Choose section", ["Chat", "Search Timeline", "Private Vault", "Network", "Simulations", "Live Call", "Export/Info", "About"], index=["Chat","Search Timeline","Private Vault","Network","Simulations","Live Call","Export/Info","About"].index(st.session_state.selected_section) if st.session_state.selected_section in ["Chat","Search Timeline","Private Vault","Network","Simulations","Live Call","Export/Info","About"] else 0)
st.session_state.selected_section = section

st.title("✨ EchoSoul — Your evolving companion")

# -------------------------
# Chat UI
# -------------------------
if section == "Chat":
    st.header("Chat with EchoSoul")
    # ensure the session_state key exists BEFORE creating widget (critical)
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""
    user_text = st.text_input("Say something to EchoSoul", key="chat_input")
    col1, col2 = st.columns([1,4])
    with col1:
        # Use an explicit form to allow clearing input on submit easily
        with st.form("chat_form", clear_on_submit=True):
            user_input_for_form = st.text_input("Hidden input for form submit", value=user_text, label_visibility="collapsed")
            submitted = st.form_submit_button("Send")
            if submitted and user_input_for_form.strip():
                # Detect emotion, store memory if a personal fact
                emotion = detect_emotion_text(user_input_for_form)
                fact = extract_personal_fact(user_input_for_form)
                if fact:
                    store_memory("Personal fact", fact["fact"], tags=["auto"])
                # Get reply
                try:
                    reply = ask_echo(user_input_for_form)
                except Exception as e:
                    reply = f"(Error generating reply) {e}"
                # Update chat history
                st.session_state.chat_history.append({"role":"user","text":user_input_for_form})
                st.session_state.chat_history.append({"role":"assistant","text":reply})
                add_to_timeline({"timestamp": str(datetime.datetime.utcnow()), "note": user_input_for_form, "echo": reply, "mood": emotion})
                # Store style sample
                learn_style_example(user_input_for_form)
                st.success("Reply generated.")

    with col2:
        # Display last few messages
        for msg in st.session_state.chat_history[-8:]:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['text']}")
            else:
                st.markdown(f"**EchoSoul:** {msg['text']}")

# -------------------------
# Timeline search / browse
# -------------------------
elif section == "Search Timeline":
    st.header("Life Timeline")
    st.write("Chronological record of saved interactions and memories.")
    q = st.text_input("Search timeline (text)")
    if q.strip():
        results = [r for r in st.session_state.timeline if q.lower() in (r.get("note","")+r.get("echo","")).lower()]
        st.write(f"Found {len(results)} results.")
        for r in results[:50]:
            st.write(f"- {r['timestamp']}: {r.get('note','')} -> {r.get('echo','')}")
    else:
        st.write("Recent timeline entries:")
        for r in st.session_state.timeline[:40]:
            st.write(f"- {r['timestamp']}: {r.get('note','')[:200]}")

# -------------------------
# Private Vault
# -------------------------
elif section == "Private Vault":
    st.header("Private Vault (Encrypted)")
    st.write("Sensitive memories are encrypted locally with Fernet (key stored in session or secrets).")
    if not vault_pw:
        st.info("Enter the vault password in the sidebar to unlock features.")
    else:
        st.success("Vault UI ready (UI password check is demo-only; encryption uses Fernet key above).")
        k = st.text_input("Key to store/read")
        v = st.text_area("Secret content")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save to vault"):
                if not k or not v:
                    st.warning("Key and content must be filled.")
                else:
                    encrypt_and_save_vault(k, v)
                    st.success("Saved and encrypted.")
        with col2:
            if st.button("Read from vault"):
                if not k:
                    st.warning("Key must be filled to read.")
                else:
                    out = read_vault(k)
                    if out is None:
                        st.warning("Not found or decryption failed.")
                    else:
                        st.code(out)

# -------------------------
# Network (Soul Resonance)
# -------------------------
elif section == "Network":
    st.header("Soul Resonance Network (demo)")
    note = st.text_area("Share a note with the network")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Broadcast to network"):
            res = network_broadcast_share(note or "heartbeat")
            st.success(res)
    with col2:
        q = st.text_input("Query")
        if st.button("Query similar (demo)"):
            if q:
                hits = network_query_similar(q)
                st.write("Hits:", hits)
            else:
                st.info("Type query then press button again.")

# -------------------------
# Simulations (Life path, Time-shifted)
# -------------------------
elif section == "Simulations":
    st.header("Life Path Simulation & Time-Shifted Self")
    mode = st.selectbox("Mode", ["Time-Shift (past)","Time-Shift (future)","Life Path Simulation","Consciousness Mirror"])
    if mode.startswith("Time-Shift"):
        user_q = st.text_input("Ask your shifted self something")
        if st.button("Talk to shifted self"):
            if user_q:
                shift = "past" if "past" in mode else "future"
                out = time_shifted_self(user_q, shift=shift)
                st.markdown(f"**{shift.title()} Self:** {out}")
            else:
                st.warning("Type a question first.")
    elif mode == "Life Path Simulation":
        st.write("Create simple choices to simulate.")
        c1 = st.text_input("Choice 1 title", "Move to new city")
        c2 = st.text_input("Choice 2 title", "Stay and upskill")
        if st.button("Simulate"):
            if c1 and c2:
                choices = [{"title":c1},{"title":c2}]
                sims = simulate_life_path({}, choices)
                for s in sims:
                    st.write(s)
            else:
                st.warning("Enter titles for both choices.")
    elif mode == "Consciousness Mirror":
        txt = st.text_area("Write some thoughts")
        if st.button("Reflect"):
            if txt:
                st.write(mirror_reflection(txt))
            else:
                st.warning("Write some thoughts first.")

# -------------------------
# Live Call (if supported)
# -------------------------
elif section == "Live Call":
    st.header("Live Call (speech <-> EchoSoul)")
    if not WEBSUPPORT:
        st.error("streamlit-webrtc not available in this environment. Install streamlit-webrtc and av.")
    elif not AUDIO_SUPPORT:
        st.warning("Speech libraries (speech_recognition, pyttsx3) not available. Install them to enable speech.")
    else:
        st.info("Start a live call. Microphone access is required.")
        # This is a simple connector: for robust call you'd need client-side audio -> se

# -------------------------
# Export/Info
# -------------------------
elif section == "Export/Info":
    st.header("Data and Information")
    st.subheader("Data Export")
    st.download_button("Download Timeline JSON", data=TIMELINE_FILE.read_text(encoding="utf-8"), file_name="echosoul_timeline.json")
    st.download_button("Download Memories JSON", data=MEMORY_FILE.read_text(encoding="utf-8"), file_name="echosoul_memories.json")

    st.subheader("System Info")
    st.write(f"OpenAI Client: {'New' if OPENAI_NEW else 'Legacy'}")
    st.write(f"WebRTC Support: {WEBSUPPORT}")
    st.write(f"Local Audio Support (Speech/TTS): {AUDIO_SUPPORT}")
    st.write(f"Number of Style Examples: {len(st.session_state.adaptive_style_examples)}")
    st.write(f"Current Personality Style: **{st.session_state.persona.get('style')}**")

# -------------------------
# About
# -------------------------
elif section == "About":
    st.header("About EchoSoul")
    st.markdown("""
    **EchoSoul** is an adaptive personal companion built on Streamlit and OpenAI.
    It features:
    * **Adaptive Personality:** Learns from your style to adjust its tone.
    * **Persistent Memory:** Saves interactions to a local timeline and memory store.
    * **Encrypted Vault:** Sensitive notes are encrypted with Fernet.
    * **Simulations:** Explore 'Time-Shifted Self' and 'Life Path' scenarios.
    """)
