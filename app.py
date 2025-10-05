import streamlit as st
import re
import json
from datetime import datetime

# =====================
# Utility Functions
# =====================

def load_data():
    try:
        with open("data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"timeline": [], "vault": []}


def save_data(data):
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)


def generate_reply(data, text, use_memories=True):
    # Dummy reply generator for now
    if use_memories and data.get("timeline"):
        return f"I hear you. Also, I remember {len(data['timeline'])} things about you."
    return f"You said: {text}"


def add_memory(data, title, content):
    data.setdefault("timeline", []).append(
        {
            "title": title,
            "content": content,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


def search_timeline(data, query):
    results = []
    for item in data.get("timeline", []):
        if query.lower() in item.get("title", "").lower() or query.lower() in item.get(
            "content", ""
        ).lower():
            results.append(item)
    return results


# Simple XOR encrypt/decrypt for demo (replace with proper crypto if needed)
def encrypt_text(password, text):
    return "".join(chr(ord(c) ^ ord(password[i % len(password)])) for i, c in enumerate(text))


def decrypt_text(password, cipher):
    try:
        return "".join(chr(ord(c) ^ ord(password[i % len(password)])) for i, c in enumerate(cipher))
    except Exception:
        return None


# =====================
# App Layout
# =====================

st.set_page_config(page_title="EchoSoul", page_icon="✨", layout="centered")
st.title("✨ EchoSoul")

# Load persistent data
data = load_data()

# Sidebar
st.sidebar.header("Settings")
vault_password = st.sidebar.text_input("Vault password", type="password")

# Tabs
tab = st.sidebar.radio(
    "Choose section", ["Chat", "Search Timeline", "Private Vault", "Export/Info"]
)

# =====================
# Chat Section
# =====================
if tab == "Chat":
    st.subheader("Chat with EchoSoul")

    user_input = st.text_input("Say something to EchoSoul", key="chat_input")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("Send"):
            if not user_input.strip():
                st.warning("Type something first.")
            else:
                reply = generate_reply(data, user_input.strip(), use_memories=True)
                st.success("Reply generated.")
                st.markdown(f"**You:** {user_input}")
                st.markdown(f"**EchoSoul:** {reply}")
                st.session_state.chat_input = ""

    with col2:
        if st.button("Save memory"):
            if not user_input.strip():
                st.warning("Type something first.")
            else:
                add_memory(data, "User memory", user_input.strip())
                save_data(data)
                st.success("Memory saved.")

    with col3:
        if st.button("Save as fact (short)"):
            text = user_input.strip()
            if not text:
                st.warning("Type something first.")
            else:
                m = re.search(r"\b(?:am|was|feel|like|love)\s+(.+)", text.lower())
                if m:
                    fact = m.group(1).strip().capitalize()
                    add_memory(data, "Fact", fact)
                    save_data(data)
                    st.success(f"Saved fact: {fact}")
                else:
                    add_memory(data, "Note", text[:200])
                    save_data(data)
                    st.success("Saved short note.")

# =====================
# Timeline Search
# =====================
elif tab == "Search Timeline":
    st.subheader("Search your timeline")
    search_query = st.text_input("Search query", key="timeline_search")

    if search_query.strip():
        results = search_timeline(data, search_query.strip())
        st.markdown(f"Found {len(results)} results (showing up to 20).")
        for r in results[:20]:
            st.markdown(f"**{r.get('title','Untitled')}** — {r.get('timestamp','')}")
            st.write(r.get("content", ""))
            st.markdown("---")

# =====================
# Vault Section
# =====================
elif tab == "Private Vault":
    st.subheader("Private Vault (encrypted)")

    if not vault_password:
        st.info("Enter a vault password in the sidebar to view or save items.")
    else:
        if not data.get("vault"):
            st.info("Vault is empty.")
        else:
            for v in data.get("vault", []):
                decrypted = decrypt_text(vault_password, v.get("cipher", ""))
                if decrypted is None:
                    st.write("*Unable to decrypt with this password.*")
                else:
                    st.write(decrypted)
                st.markdown("---")

        st.markdown("### Add a new vault item")
        vt = st.text_input("Title", key="vt")
        vc = st.text_area("Secret content", key="vc")
        if st.button("Save to vault"):
            if not vc.strip():
                st.warning("Secret content cannot be empty.")
            else:
                cipher = encrypt_text(vault_password, vc.strip())
                data.setdefault("vault", []).append(
                    {"title": vt or "Vault item", "cipher": cipher}
                )
                save_data(data)
                st.success("Saved to vault (encrypted).")

# =====================
# Export / Info
# =====================
elif tab == "Export/Info":
    st.subheader("Legacy & Export")
    st.json(data)
