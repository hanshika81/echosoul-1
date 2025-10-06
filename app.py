import os
import json
import hashlib
import base64
from cryptography.fernet import Fernet
import streamlit as st
from openai import OpenAI
from datetime import datetime

# --------------------
# OpenAI Setup
# --------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --------------------
# Utility functions
# --------------------
def generate_key(password: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

def encrypt_text(password: str, text: str) -> str:
    try:
        fernet = Fernet(generate_key(password))
        return fernet.encrypt(text.encode()).decode()
    except Exception:
        return None

def decrypt_text(password: str, token: str) -> str:
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
    context = ""
    if use_memories and "timeline" in data and len(data["timeline"]) > 0:
        last_items = data["timeline"][-5:]
        context = "\n".join([f"{it['title']}: {it['content']}" for it in last_items])

    messages = [
        {"role": "system", "content": "You are EchoSoul, an AI memory companion."},
        {"role": "system", "content": f"Memory context:\n{context}"},
        {"role": "user", "content": user_input}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # use gpt-4 or gpt-3.5-turbo if preferred
            messages=messages,
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
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

vault_password = st.sidebar.text_input("Vault password", type="password")
section = st.sidebar.radio("Choose section", ["Chat", "Search Timeline", "Private Vault", "Export/Info"])

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

            data["timeline"].append({
                "title": f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "content": f"You: {user_input}\nEchoSoul: {reply}"
            })
            save_data(data)

            st.success("Reply generated.")
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**EchoSoul:** {reply}")

            # Clear input properly
            st.session_state.update({"chat_input": ""})
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
