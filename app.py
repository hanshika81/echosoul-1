# app.py
"""
EchoSoul - Streamlit app main file.

Design goals:
 - Use new OpenAI client API (openai.OpenAI)
 - Graceful fallback if optional heavy libs (spaCy, textblob, nltk, streamlit-webrtc, cryptography) are missing
 - Provide: chat, chat history, simple "brain mimic" keyword extraction, vault (optional encryption),
   audio upload -> transcription (optional), environment secret reading, and helpful messages.
"""

import os
import re
import time
import json
from typing import List, Dict, Optional

import streamlit as st

# try to use new OpenAI client interface
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# optional libs that might be heavy to install on free hosts
try:
    import spacy  # optional, for better keyword extraction if available
    _HAS_SPACY = True
except Exception:
    _HAS_SPACY = False

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except Exception:
    _HAS_TEXTBLOB = False

try:
    import nltk
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    from cryptography.fernet import Fernet
    _HAS_CRYPTO = True
except Exception:
    _HAS_CRYPTO = False

# streamlit-webrtc is optional for browser-based mic; we'll gracefully fall back to file upload
try:
    import streamlit_webrtc  # type: ignore
    _HAS_STREAMLIT_WEBRTC = True
except Exception:
    _HAS_STREAMLIT_WEBRTC = False

# helper: load .env locally if present (but don't rely on it in production)
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# ------------------------------------------------------------
# Config & secrets
# ------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AGORA_APP_ID = os.getenv("AGORA_APP_ID")
AGORA_APP_CERTIFICATE = os.getenv("AGORA_APP_CERTIFICATE")

# Create OpenAI client if possible
openai_client = None
if _HAS_OPENAI and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None

# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------
def safe_extract_keywords(text: str, top_n: int = 6) -> List[str]:
    """
    Extract keywords from text with best available method:
      - spacy (if installed)
      - textblob noun phrases (if installed)
      - fallback: short unique tokens with simple heuristics
    """
    text = (text or "").strip()
    if not text:
        return []

    # Use spaCy if available and has a model loaded
    if _HAS_SPACY:
        try:
            # we don't call spacy.load("en_core_web_sm") automatically because the model
            # may not be installed. Try to use default spaCy pipeline.
            nlp = spacy.blank("en") if "en_core_web_sm" not in spacy.util.get_installed_models() else spacy.load("en_core_web_sm")
            doc = nlp(text)
            candidates = [chunk.text.lower() for chunk in doc.noun_chunks]
            # fallback to tokens if noun_chunks empty
            if not candidates:
                candidates = [t.text.lower() for t in doc if not t.is_stop and t.is_alpha]
            # dedupe preserving order
            seen = set()
            out = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    out.append(c)
            return out[:top_n]
        except Exception:
            pass

    # Use TextBlob noun phrases
    if _HAS_TEXTBLOB:
        try:
            tb = TextBlob(text)
            phrases = [p.lower() for p in tb.noun_phrases]
            return phrases[:top_n]
        except Exception:
            pass

    # Simple regex/token fallback
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    # sort by frequency then length
    sorted_tokens = sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))
    return [t for t, _ in sorted_tokens[:top_n]]

def openai_chat_reply(user_text: str, system_prompt: str = "You are EchoSoul, an adaptive AI companion.", history: List[Dict] = None, model: str = "gpt-3.5-turbo") -> str:
    """
    Use the new OpenAI client (openai.OpenAI) chat completions API format.
    Falls back gracefully if client not available.
    """
    if not user_text:
        return ""
    if openai_client is None:
        return "OpenAI client not configured. Please set OPENAI_API_KEY in your environment."

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    # include history as previous user/assistant messages if given
    if history:
        for m in history:
            # each history item should be a dict with role/content
            if m.get("role") and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_text})

    try:
        # new client: client.chat.completions.create(...)
        response = openai_client.chat.completions.create(model=model, messages=messages)
        # response.choices[0].message.content
        choice = response.choices[0]
        msg = getattr(choice, "message", None)
        if msg:
            return getattr(msg, "content", str(choice))
        # fallback if attributes different
        return str(choice)
    except Exception as e:
        # return friendly message to user
        return f"Error calling OpenAI chat completion: {e}"

# Simple vault encryption (optional)
def make_fernet_key_from_password(password: str) -> bytes:
    # Very simple key derivation for demo only: do NOT use for production secret management
    # We hash and use base64 urlsafe as Fernet key
    import hashlib, base64
    h = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(h)

def encrypt_text(plain: str, password: str) -> Optional[str]:
    if not _HAS_CRYPTO:
        return None
    try:
        key = make_fernet_key_from_password(password)
        f = Fernet(key)
        token = f.encrypt(plain.encode("utf-8"))
        return token.decode("utf-8")
    except Exception:
        return None

def decrypt_text(token_str: str, password: str) -> Optional[str]:
    if not _HAS_CRYPTO:
        return None
    try:
        key = make_fernet_key_from_password(password)
        f = Fernet(key)
        raw = f.decrypt(token_str.encode("utf-8"))
        return raw.decode("utf-8")
    except Exception:
        return None

# Audio transcription via OpenAI audio endpoint (best-effort)
def transcribe_audio_file(file_bytes: bytes, filename: str = "upload.wav", model: str = "whisper-1") -> str:
    """
    Transcribe audio bytes using OpenAI audio endpoint if available.
    This uses the new client model; if unavailable, returns a helpful message.
    """
    if openai_client is None:
        return "OpenAI client not available to transcribe audio. Set OPENAI_API_KEY."

    try:
        # The new client exposes audio transcriptions on client.audio.transcriptions.create
        # This will only work if the installed openai package exposes that API.
        # We'll try best-effort and catch errors.
        import io
        audio_file = io.BytesIO(file_bytes)
        # Some installed clients accept 'file' named parameter; this may vary across versions.
        resp = openai_client.audio.transcriptions.create(model=model, file=audio_file)
        # Many responses have text
        text = getattr(resp, "text", None) or resp.get("text") if isinstance(resp, dict) else None
        if text:
            return text
        # fallback: try to stringify resp
        return str(resp)
    except Exception as e:
        return f"Audio transcription failed: {e}"

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("🌙 EchoSoul — your adaptive AI companion")

# Sidebar: mode, environment debug toggle
with st.sidebar:
    st.header("EchoSoul")
    mode = st.radio("Mode", ["Chat", "Chat history", "Life timeline", "Vault", "Call", "About"], index=0)
    show_env = st.checkbox("Show env debug (hide keys)", value=False)
    if show_env:
        st.subheader("Environment (masked)")
        def mask(v):
            if not v:
                return "<not set>"
            s = str(v)
            if len(s) <= 6:
                return "*" * len(s)
            return s[:3] + "*" * (len(s) - 6) + s[-3:]
        st.write("OPENAI_API_KEY:", mask(OPENAI_API_KEY))
        st.write("AGORA_APP_ID:", mask(AGORA_APP_ID))
        st.write("AGORA_APP_CERT:", mask(AGORA_APP_CERT))

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts {"role": "user"/"assistant", "content": "..."}
if "vault" not in st.session_state:
    st.session_state.vault = {}  # simple key->encrypted value
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "You are EchoSoul, an adaptive AI companion."

# Helper UI bits
def append_history(role: str, content: str):
    st.session_state.history.append({"role": role, "content": content})

# Main: Chat
if mode == "Chat":
    st.subheader("💬 Chat with EchoSoul")
    user_input = st.text_area("Your message", height=150, value="", placeholder="Share your thoughts...")
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("Send"):
            if not user_input.strip():
                st.warning("Please write a message before sending.")
            else:
                append_history("user", user_input)
                with st.spinner("Thinking..."):
                    reply = openai_chat_reply(user_input, system_prompt=st.session_state.system_prompt, history=st.session_state.history[-10:])
                append_history("assistant", reply)
                st.experimental_rerun()
    with c2:
        st.markdown("**Conversation**")
        for m in st.session_state.history:
            if m["role"] == "user":
                st.markdown(f"**You:** {m['content']}")
            else:
                st.markdown(f"**EchoSoul:** {m['content']}")

# Chat history view
elif mode == "Chat history":
    st.subheader("🗂️ Chat history")
    st.write("Saved messages in session (not persisted).")
    for idx, m in enumerate(st.session_state.history):
        st.write(f"{idx+1}. [{m['role']}] {m['content'][:400]}")

# Life timeline (simple placeholder)
elif mode == "Life timeline":
    st.subheader("🕰️ Life timeline")
    st.info("This feature would show memory timeline items. For demo it uses chat 'assistant' messages as timeline items.")
    timeline_items = [m for m in st.session_state.history if m["role"] == "assistant"]
    for t in timeline_items[-20:]:
        st.write(f"- {t['content']}")

# Vault: store encrypted notes (optional: cryptography)
elif mode == "Vault":
    st.subheader("🔐 Vault (encrypted notes)")
    if not _HAS_CRYPTO:
        st.warning("Vault requires the 'cryptography' package. The app can store plaintext in session but encryption isn't available.")
    pwd = st.text_input("Vault password (used to encrypt/decrypt) — keep safe", type="password")
    key = st.text_input("Note key (identifier)", value="")
    note = st.text_area("Note to store", value="", height=120)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save note"):
            if not key:
                st.error("Please enter a note key.")
            else:
                if _HAS_CRYPTO and pwd:
                    enc = encrypt_text(note, pwd)
                    if enc is None:
                        st.error("Encryption failed.")
                    else:
                        st.session_state.vault[key] = {"enc": enc}
                        st.success(f"Saved encrypted note as '{key}'")
                else:
                    # fallback store plaintext
                    st.session_state.vault[key] = {"plain": note}
                    st.success(f"Saved plaintext note as '{key}'")
    with col2:
        if st.button("List notes"):
            st.write(list(st.session_state.vault.keys()))
        if st.button("Clear vault (session only)"):
            st.session_state.vault = {}
            st.success("Vault cleared (session).")
    st.markdown("---")
    st.write("Retrieve note")
    retrieve_key = st.text_input("Key to retrieve")
    if st.button("Get note"):
        if retrieve_key in st.session_state.vault:
            entry = st.session_state.vault[retrieve_key]
            if "enc" in entry:
                if not pwd:
                    st.error("This note is encrypted. Provide the vault password to decrypt.")
                else:
                    dec = decrypt_text(entry["enc"], pwd)
                    if dec is None:
                        st.error("Decryption failed (wrong password?).")
                    else:
                        st.text_area("Decrypted note", dec, height=200)
            else:
                st.text_area("Note (plaintext)", entry.get("plain", ""), height=200)
        else:
            st.warning("No note by that key in session vault.")

# Call (voice) mode: best-effort
elif mode == "Call":
    st.subheader("📞 Live Voice Chat")
    st.info("Grant microphone permission in your browser. If WebRTC support isn't available, you can upload audio files for transcription.")
    mic_supported = _HAS_STREAMLIT_WEBRTC
    if mic_supported:
        st.success("streamlit-webrtc available: browser-based mic capture may work.")
    else:
        st.warning("streamlit-webrtc not installed — fallback to upload.")

    st.markdown("**Upload audio file to transcribe**")
    uploaded = st.file_uploader("Upload WAV/MP3/OGG", type=["wav", "mp3", "ogg", "m4a"])
    if uploaded is not None:
        try:
            b = uploaded.read()
            st.info("Sending audio to transcription (best-effort). This may take a few seconds.")
            transcription = transcribe_audio_file(b, filename=uploaded.name)
            st.success("Transcription result:")
            st.write(transcription)
            # optionally send transcription to chat
            if st.button("Send transcription as user message"):
                append_history("user", transcription)
                with st.spinner("Generating reply..."):
                    reply = openai_chat_reply(transcription, system_prompt=st.session_state.system_prompt, history=st.session_state.history[-10:])
                append_history("assistant", reply)
                st.success("Reply saved to session conversation.")
        except Exception as e:
            st.error(f"Failed to read or transcribe uploaded file: {e}")

    st.markdown("**Live mic / WebRTC (if supported)**")
    if _HAS_STREAMLIT_WEBRTC:
        st.info("If streamlit-webrtc is installed we could run a small recorder. For safety and stability on free hosts we avoid creating a persistent webRTC server here.")
        st.write("If you need fully interactive WebRTC voice, consider using a dedicated client-side implementation or deploy to a paid instance.")
    else:
        st.info("Install `streamlit-webrtc` locally to test browser mic; Render may not support persistent webRTC easily on free plans.")

# About
elif mode == "About":
    st.header("About EchoSoul")
    st.markdown(
        """
        EchoSoul is a demo adaptive AI companion built with Streamlit and OpenAI.
        This app:
        - Uses the new OpenAI client API (`openai>=1.0.0`) via `OpenAI`.
        - Degrades gracefully if optional heavy libraries are missing.
        - Requires `OPENAI_API_KEY` to be set as an environment variable (in Render / Railway UI).
        """
    )
    st.markdown("**Troubleshooting tips**")
    st.markdown(
        "- If you see `You tried to access openai.ChatCompletion` error: update your code to the new OpenAI client (this app uses it).\n"
        "- If you see `Permission denied` on call: grant mic permissions in the browser (click lock icon near the URL).\n"
        "- If audio or packages fail during deploy: ensure your `requirements.txt` pins versions compatible with your Python runtime."
    )

# Brain mimic / keyword diagnostics panel (below main UI)
st.sidebar.markdown("---")
st.sidebar.subheader("Brain mimic tools")
sample_text = st.sidebar.text_area("Sample text for keyword extraction (brain mimic)", value="I love hiking near the sea and writing songs about memories.", height=80)
if st.sidebar.button("Extract keywords from sample"):
    kw = safe_extract_keywords(sample_text, top_n=8)
    st.sidebar.write("Keywords:", kw)

# small footer info
st.markdown("---")
st.caption("EchoSoul — demo. Keep your OPENAI_API_KEY secret. This app stores session data in memory only and does not persist chat history by default.")
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    st.run()
