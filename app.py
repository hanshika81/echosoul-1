import streamlit as st
import json
import os
from datetime import datetime
from cryptography.fernet import Fernet
import openai

# ========================
# OpenAI Setup
# ========================
openai.api_key = st.secrets["OPENAI_API_KEY"]

def generate_reply(data, user_input, use_memories=True):
    """
    Generate a reply from EchoSoul using OpenAI's GPT model.
    If use_memories=True, include recent timeline/context in the prompt.
    """
    context = ""
    if use_memories and "timeline" in data:
        last_items = data["timeline"][-5:]  # last 5 items for context
        context = "\n".join([f"{it['title']}: {it['content']}" for it in last_items])

    prompt = f"""
    You are EchoSoul, an AI memory companion.
    Here is your memory context:
    {context}

    User: {user_input}
    EchoSoul:"""

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].text.strip()


# ========================
# Encryption helpers
# ========================
def get_key_from_password(password: str) -> bytes:
    """Derive Fernet key from password"""
    return Fernet.generate_key()  # Simplified (replace with PBKDF2 in production)

def encrypt_text(password: str, text: str) -> str:
    key = get_key_from_password(password)
    fernet = Fernet(key)
    return fernet.encrypt(text.encode()).decode()

def decrypt_text(password: str, token: str) -> str:
    try:
        key = get_key_from_password(password)
        fernet = Fernet(key)
        return fernet.decrypt(token.encode()).decode()
    except Exception:
        return None


# ========================
# Data helpers
# ========================
DATA_FILE = "data.json"

def load_data():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return {"timeline": [], "vault": []}

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ========================
# Search helpers
# ========================
def search_timeline(data, query):
    results = []
    for item in data.get("timeline", []):
        if query.lower() in item["content"].lower() or query.lower() in item["title"].lower():
            results.append(item)
    return results


# ========================
# Streamlit UI
# ========================
st.set_page_config(page_title="EchoSoul", layout="wide")
st.title("✨ EchoSoul")

# Sidebar
st.sidebar.header("Settings")
vault_password = st.sidebar.text_input("Vault password", type="password")
tab = st.sidebar.radio("Choose section", ["Chat", "Search Timeline", "Private Vault", "Export/Info"])

# Load data
data = load_data()

# ========================
# Chat Section
# ========================
if tab == "Chat":
    st.header("Chat with EchoSoul")
    user_input = st.text_input("Say something to EchoSoul", key="chat_input")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Send"):
            if not user_input.strip():
                st.warning("Type something first.")
            else:
                reply = generate_reply(data, user_input.strip(), use_memories=True)
                st.success("Reply generated.")

                # Show reply inline
                st.markdown(f"**You:** {user_input}")
                st.markdown(f"**EchoSoul:** {reply}")

                # Save to timeline
                data.setdefault("timeline", []).append({
                    "title": "Chat",
                    "content": f"You: {user_input}\nEchoSoul: {reply}",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_data(data)


# ========================
# Timeline Search
# ========================
elif tab == "Search Timeline":
    st.header("🔍 Search Timeline")
    search_query = st.text_input("Search query", key="timeline_search")
    if search_query.strip():
        results = search_timeline(data, search_query.strip())
        st.markdown(f"Found {len(results)} results (showing up to 20).")
        for r in results[:20]:
            st.markdown(f"**{r['title']}** — {r['timestamp']}")
            st.write(r["content"])
            st.markdown("---")


# ========================
# Private Vault
# ========================
elif tab == "Private Vault":
    st.header("🔐 Private Vault")

    vt = st.text_input("Title for vault item", key="vt")
    vc = st.text_area("Secret content", key="vc")

    if st.button("Save to vault"):
        if not vault_password:
            st.warning("Set a vault password in the sidebar first.")
        elif not vc.strip():
            st.warning("Secret content cannot be empty.")
        else:
            cipher = encrypt_text(vault_password, vc.strip())
            data.setdefault("vault", []).append({
                "title": vt or "Vault item",
                "cipher": cipher,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            save_data(data)
            st.success("Saved to vault (encrypted).")

    if st.button("Show vault items"):
        if not vault_password:
            st.warning("Enter vault password in sidebar to decrypt.")
        else:
            for v in data.get("vault", []):
                decrypted = decrypt_text(vault_password, v.get("cipher", ""))
                if decrypted is None:
                    st.write("❌ Unable to decrypt (wrong password?)")
                else:
                    st.markdown(f"**{v['title']}** — {v['timestamp']}")
                    st.write(decrypted)
                    st.markdown("---")


# ========================
# Export & Info
# ========================
elif tab == "Export/Info":
    st.header("📦 Export / Info")
    st.write("Export your data or produce a human-readable snapshot.")

    # Download JSON
    if st.download_button("Download full export (JSON)", json.dumps(data, indent=2), file_name="echosoul_export.json"):
        st.success("Download started.")

    # Human-readable snapshot
    legacy_lines = [f"[{it['timestamp']}] {it['title']}: {it['content']}" for it in data.get("timeline", [])]
    legacy_text = "\n\n".join(legacy_lines)
    st.text_area("Legacy snapshot", value=legacy_text, height=300)

    st.markdown("---")
    st.header("ℹ️ About EchoSoul")
    st.write("EchoSoul runs a GPT-powered memory companion. It stores chats in a timeline, encrypts private notes, and lets you search or export everything.")
