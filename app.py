import os
import json
import hashlib
import base64
from cryptography.fernet import Fernet
import streamlit as st
import openai
from datetime import datetime

# --------------------
# OpenAI Setup
# --------------------
openai.api_key = os.getenv("OPENAI_API_KEY")

# --------------------
# Utility functions
# --------------------
def generate_key(password: str) -> bytes:
    """Generate a Fernet key from password."""
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

def encrypt_text(password: str, text: str) -> str:
    """Encrypt text using password."""
    try:
        fernet = Fernet(generate_key(password))
        return fernet.encrypt(text.encode()).decode()
    except Exception:
        return None

def decrypt_text(password: str, token: str) -> str:
    """Decrypt text using password."""
    try:
        fernet = Fernet(generate_key(password))
        return fernet.decrypt(token.encode()).decode()
    except Exception:
        return None

def load_data():
    if os.path.exists("data.json"):
        with open("data.json", "r") as f:
            return json.load(f)
    return {"timeline": [], "vault": []}

def save_data(data):
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)

# --------------------
# AI Reply Generator
# --------------------
def generate_reply(data, user_input, use_memories=True):
    """Generate reply using OpenAI GPT, with optional memory context."""
    context = ""
    if use_memories and "timeline" in data and len(data["timeline"]) > 0:
        last_items = data["timeline"][-5:]  # last 5 items
        context = "\n".join([f"{it['title']}: {it['content']}" for it in last_items])

    prompt = f"""
    You are EchoSoul, an AI memory companion.
    Memory context:
    {context}

    User: {user_input}
    EchoSoul:"""

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"(Error: {e})"

def search_timeline(data, query: str):
    results = []
    for item in data.get("timeline", []):
        if query.lower() in item["title"].lower() or query.lower() in item["content"].lower():
            results.append(item)
    return results

# --------------------
# Streamlit UI
# --------------------
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("✨ EchoSoul")
st.sidebar.header("Settings")

# Sidebar
vault_password = st.sidebar.text_input("Vault password", type="password")
section = st.sidebar.radio("Choose section", ["Chat", "Search Timeline", "Private Vault", "Export/Info"])

# Load data
data = load_data()

# --------------------
# Sections
# --------------------
if section == "Chat":
    st.subheader("Chat with EchoSoul")

    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    user_input = st.text_input("Say something to EchoSoul", key="chat_input")

    if st.button("Send"):
        if not user_input.strip():
            st.warning("Type something first.")
        else:
            reply = generate_reply(data, user_input.strip(), use_memories=True)

            # Save to timeline
            data["timeline"].append({
                "title": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "content": f"You: {user_input}\nEchoSoul: {reply}"
            })
            save_data(data)

            st.success("Reply generated.")
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**EchoSoul:** {reply}")

            # Clear input safely
            st.session_state.chat_input = ""

elif section == "Search Timeline":
    st.subheader("Search your timeline")
    search_query = st.text_input("Search query", key="timeline_search")
    if search_query.strip():
        results = search_timeline(data, search_query.strip())
        st.markdown(f"Found **{len(results)}** results")
        for r in results[:20]:
            st.markdown(f"**{r['title']}**")
            st.write(r["content"])
            st.markdown("---")

elif section == "Private Vault":
    st.subheader("Private Vault")
    if vault_password:
        vt = st.text_input("Title for vault item", key="vt")
        vc = st.text_area("Secret content", key="vc")
        if st.button("Save to vault"):
            encrypted = encrypt_text(vault_password, vc)
            if encrypted:
                data["vault"].append({"title": vt, "cipher": encrypted})
                save_data(data)
                st.success("Saved to vault")
            else:
                st.error("Encryption failed.")

        st.markdown("### Vault Items")
        for v in data.get("vault", []):
            decrypted = decrypt_text(vault_password, v.get("cipher", ""))
            if decrypted is None:
                st.write(f"**{v['title']}** - 🔒 Unable to decrypt")
            else:
                st.write(f"**{v['title']}** - {decrypted}")
    else:
        st.warning("Enter vault password in sidebar to unlock.")

elif section == "Export/Info":
    st.subheader("Export Data / Info")
    st.download_button("Download timeline (JSON)", json.dumps(data["timeline"], indent=2), file_name="timeline.json")
    st.download_button("Download vault (JSON)", json.dumps(data["vault"], indent=2), file_name="vault.json")
    st.info("EchoSoul stores memories, vault items, and lets you chat with AI. Data is saved locally in `data.json`.")
