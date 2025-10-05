# app.py — Full EchoSoul (GPT brain + timeline + vault + adaptive persona + export)
import streamlit as st
import os, json, hashlib, base64, datetime, re, typing
from openai import OpenAI
import os
from openai import OpenAI
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
# Try to import cryptography Fernet; fallback to a lightweight XOR cipher
try:
    from cryptography.fernet import Fernet, InvalidToken
    CRYPTO_AVAILABLE = True
except Exception:
    CRYPTO_AVAILABLE = False

# Initialize OpenAI (key is read from Streamlit secrets)
import os
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Data file (in-app directory; streamlit cloud allowed)
DATA_FILE = "echosoul_data.json"

# -------------------------
# Utilities: storage / timestamps
# -------------------------
def ts_now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def default_data():
    return {
        "profile": {
            "name": "",
            "created": ts_now(),
            "persona": {"tone": "friendly", "style": "casual"}
        },
        "timeline": [],
        "vault": [],
        "conversations": []
    }

def load_data() -> dict:
    if not os.path.exists(DATA_FILE):
        d = default_data()
        save_data(d)
        return d
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # corrupt or unreadable: reset to default
        d = default_data()
        save_data(d)
        return d

def save_data(data: dict):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# -------------------------
# Encryption utilities (vault)
# -------------------------

# Fernet helper (generate a wrapping key using password)
def _fernet_from_password(password: str) -> Fernet:
    # Derive deterministic 32-byte key from password using SHA256, then base64
    import hashlib, base64
    key = hashlib.sha256(password.encode("utf-8")).digest()
    key_b64 = base64.urlsafe_b64encode(key)
    return Fernet(key_b64)

def encrypt_text(password: str, plaintext: str) -> str:
    if CRYPTO_AVAILABLE:
        f = _fernet_from_password(password)
        token = f.encrypt(plaintext.encode("utf-8"))
        return base64.b64encode(token).decode("utf-8")  # store base64 to be safe
    # fallback XOR-ish (not secure) — only for demo
    b = plaintext.encode("utf-8")
    key = hashlib.sha256(password.encode("utf-8")).digest()
    out = bytes([b[i] ^ key[i % len(key)] for i in range(len(b))])
    return base64.b64encode(out).decode("utf-8")

def decrypt_text(password: str, cipher_b64: str) -> typing.Optional[str]:
    try:
        raw = base64.b64decode(cipher_b64.encode("utf-8"))
    except Exception:
        return None
    if CRYPTO_AVAILABLE:
        try:
            f = _fernet_from_password(password)
            pt = f.decrypt(raw)
            return pt.decode("utf-8")
        except Exception:
            return None
    # fallback XOR decode
    key = hashlib.sha256(password.encode("utf-8")).digest()
    try:
        out = bytes([raw[i] ^ key[i % len(key)] for i in range(len(raw))])
        return out.decode("utf-8")
    except Exception:
        return None

# -------------------------
# Simple sentiment (heuristic) — used to adapt persona
# -------------------------
POS_WORDS = set("good great happy love excellent amazing wonderful nice grateful fun delighted excited calm optimistic".split())
NEG_WORDS = set("bad sad angry depressed unhappy terrible awful hate lonely anxious stressed worried frustrated".split())

def sentiment_score(text: str) -> float:
    toks = re.findall(r"\w+", text.lower())
    pos = sum(1 for t in toks if t in POS_WORDS)
    neg = sum(1 for t in toks if t in NEG_WORDS)
    score = pos - neg
    norm = score / max(1, len(toks))
    return norm

def sentiment_label(score: float) -> str:
    if score > 0.06:
        return "positive"
    if score < -0.06:
        return "negative"
    return "neutral"

def update_persona_based_on_sentiment(data: dict, score: float):
    if not data.get("profile"):
        return
    if score < -0.06:
        data["profile"]["persona"]["tone"] = "empathetic"
    elif score > 0.06:
        data["profile"]["persona"]["tone"] = "energetic"
    else:
        data["profile"]["persona"]["tone"] = "friendly"
    save_data(data)

# -------------------------
# Memory helpers
# -------------------------
def add_memory(data: dict, title: str, content: str):
    item = {
        "id": hashlib.sha1((title + content + ts_now()).encode("utf-8")).hexdigest(),
        "title": title,
        "content": content,
        "timestamp": ts_now()
    }
    data["timeline"].append(item)
    save_data(data)
    return item

def search_timeline(data: dict, query: str, limit: int = 20):
    q = query.lower()
    results = [m for m in reversed(data["timeline"]) if q in (m["title"] + " " + m["content"]).lower()]
    return results[:limit]

def find_relevant_memories(data: dict, text: str, limit: int = 3):
    txt = text.lower()
    found = []
    for item in reversed(data["timeline"]):
        if any(w in txt for w in re.findall(r"\w+", item["content"].lower())) or any(w in txt for w in re.findall(r"\w+", item["title"].lower())):
            found.append(item)
            if len(found) >= limit:
                break
    return found

# -------------------------
# GPT integration (generate reply)
# -------------------------
def generate_reply(data: dict, user_msg: str, use_memories: bool = True) -> str:
    # sentiment + persona update
    s = sentiment_score(user_msg)
    update_persona_based_on_sentiment(data, s)
    persona_tone = data["profile"].get("persona", {}).get("tone", "friendly")

    # build memory context
    mem_items = data.get("timeline", [])[-5:] if use_memories else []
    mem_text = "\n".join([f"- {m['title']}: {m['content']}" for m in mem_items]) or "No memories yet."

    # detect if user asks to "act like me"
    low = user_msg.lower()
    act_like_me = any(phrase in low for phrase in ["act like me", "be me", "reply like me", "speak like me", "roleplay as me"])

    # system prompt
    system_prompt_lines = [
        f"You are EchoSoul — a warm, evolving personal companion for {data['profile'].get('name', 'User')}.",
        f"Personality tone: {persona_tone}. Style: {data['profile'].get('persona',{}).get('style','casual')}.",
        "You should be empathetic, concise, and reflective. Use the user's name when appropriate.",
        "",
        "Memory context (most recent items):",
        mem_text,
        "",
        "Behavior rules:",
        "- If the user asks you to 'act like me' or similar, roleplay as the user and answer in first-person as if you are them.",
        "- If the user asks a direct question, answer helpfully and clearly.",
        "- Ask gentle follow-up questions when a response could benefit from more detail.",
        "- Keep responses relatively brief (1-6 sentences) unless asked for longer."
    ]
    system_prompt = "\n".join(system_prompt_lines)

    # user messages: include special instruction if acting like user
    user_content = user_msg
    if act_like_me:
        user_content = "(User asked: act like them.) " + user_msg

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=500
        )
        # extract text
        reply = resp.choices[0].message.content.strip()
    except Exception as e:
        # fallback simple reply if API errors
        reply = "Sorry — I'm having trouble connecting to my AI brain right now. I still remember what you said: " + user_msg

    # save conversation
    data.setdefault("conversations", []).append({"user": user_msg, "bot": reply, "ts": ts_now()})
    save_data(data)
    return reply

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="EchoSoul", layout="centered")

st.title("EchoSoul — Your evolving companion")

data = load_data()

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    name = st.text_input("Your name", value=data["profile"].get("name", ""))
    if st.button("Save profile name"):
        data["profile"]["name"] = name.strip() or data["profile"].get("name","")
        save_data(data)
        st.success("Profile saved.")
    st.markdown("---")
    st.markdown("Vault settings (password not saved):")
    vault_password = st.text_input("Vault password (used to encrypt/decrypt)", type="password")
    if CRYPTO_AVAILABLE:
        st.caption("Using secure Fernet encryption.")
    else:
        st.caption("Cryptography not available; using demo fallback (not secure).")
    st.markdown("---")
    st.checkbox("Enable adaptive persona (sentiment influences tone)", value=True, key="adaptive_toggle")
    st.markdown("---")
    st.write("Data file:", DATA_FILE)
    st.markdown("If you want a fresh start, you can reset data below.")
    if st.button("Reset all EchoSoul data"):
        save_data(default_data())
        st.success("Data reset. Refresh the app.")

# Main tabs
tab = st.radio("", ["Chat", "Life Timeline", "Private Vault", "Legacy & Export", "About"])

if tab == "Chat":
    st.subheader("Chat with EchoSoul")
    # show name
    st.markdown(f"**You:** {data['profile'].get('name','(no name)')}")

    # conversation display
    convs = data.get("conversations", [])
    for c in convs[-30:]:
        st.markdown(f"**You:** {c['user']}")
        st.markdown(f"**EchoSoul:** {c['bot']}")
        st.markdown("---")

    user_input = st.text_input("Say something to EchoSoul", key="chat_input")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    if st.button("Send"):
        if not user_input.strip():
            st.warning("Type something first.")
        else:
            reply = generate_reply(data, user_input.strip(), use_memories=True)
            st.success("Reply generated.")
            # show new reply inline
            st.markdown(f"**You:** {user_input}")
            st.markdown(f"**EchoSoul:** {reply}")
            st.session_state.chat_input = ""   # 👈 clears the box after sending
    with col2:
        if st.button("Add to timeline"):
            if not user_input.strip():
                st.warning("Type the memory content in the box first.")
            else:
                add_memory(data, "User added memory", user_input.strip())
                st.success("Memory saved to timeline.")
    with col3:
        if st.button("Save as fact (short)"):
            text = user_input.strip()
            if not text:
                st.warning("Type something first.")
            else:
                # detect "I am ...", "I like ..."
                m = re.search(r"\bi (?:am|was|feel|like|love)\s+(.+)", text.lower())
                if m:
                    fact = m.group(1).strip().capitalize()
                    add_memory(data, "Personal note", fact)
                    st.success(f"Saved fact: {fact}")
                else:
                    add_memory(data, "Short note", text[:200])
                    st.success("Saved short note.")

if tab == "Life Timeline":
    st.subheader("Life Timeline — add, view, search")
    st.markdown("Add a memory or browse/search your timeline.")
    with st.form("add_memory_form"):
        ttitle = st.text_input("Title", value="Memory")
        tcontent = st.text_area("Content")
        submitted = st.form_submit_button("Save Memory")
        if submitted:
            if not tcontent.strip():
                st.warning("Content cannot be empty.")
            else:
                add_memory(data, ttitle.strip() or "Memory", tcontent.strip())
                st.success("Memory saved.")

    st.markdown("### Recent memories")
    for item in sorted(data["timeline"], key=lambda x: x["timestamp"], reverse=True)[:30]:
        st.markdown(f"**{item['title']}** — {item['timestamp']}")
        st.write(item["content"])
        st.markdown("---")

    st.markdown("### Search timeline")
    q = st.text_input("Search query", key="timeline_search")
    if q.strip():
        results = search_timeline(data, q.strip())
        st.markdown(f"Found {len(results)} results (showing up to 20).")
        for r in results:
            st.markdown(f"**{r['title']}** — {r['timestamp']}")
            st.write(r["content"])
            st.markdown("---")

if tab == "Private Vault":
    st.subheader("Private Vault (encrypted notes)")
    st.write("Vault items remain encrypted. You must provide the same password to decrypt them.")
    if not data.get("vault"):
        st.info("Vault is empty.")

    # show vault items (titles only, decrypt when password provided)
    for idx, v in enumerate(data.get("vault", [])):
        st.markdown(f"**{v.get('title','Vault item')}** — {v.get('timestamp')}")
        if vault_password:
            decrypted = decrypt_text(vault_password, v.get("cipher",""))
            if decrypted is None:
                st.write("*Unable to decrypt with this password.*")
            else:
                st.write(decrypted)
        else:
            st.write("*Provide password in sidebar to view.*")
        st.markdown("---")

    st.markdown("Add a new vault item")
    vt = st.text_input("Title for vault item", key="vt")
    vc = st.text_area("Secret content", key="vc")
    if st.button("Save to vault"):
        if not vault_password:
            st.warning("Set a vault password in the sidebar first.")
        elif not vc.strip():
            st.warning("Secret content cannot be empty.")
        else:
            cipher = encrypt_text(vault_password, vc.strip())
            data.setdefault("vault", []).append({"title": vt or "Vault item", "cipher": cipher, "timestamp": ts_now()})
            save_data(data)
            st.success("Saved to vault (encrypted).")

if tab == "Legacy & Export":
    st.subheader("Legacy & Export")
    st.write("Export your data or produce a human-readable snapshot.")
    # Download full JSON
    if st.button("Download full export (JSON)"):
        st.download_button("Click to download JSON", json.dumps(data, indent=2), f"echosoul_export_{datetime.datetime.utcnow().date()}.json", "application/json")
    st.markdown("---")
    st.write("Legacy snapshot (human-readable timeline):")
    legacy_lines = [f"{it['timestamp']}: {it['title']} — {it['content']}" for it in data.get("timeline",[])]
    legacy_text = "\n\n".join(legacy_lines)
    st.text_area("Legacy snapshot", value=legacy_text, height=300)

if tab == "About":
    st.header("About EchoSoul (full version)")
    st.write("This app runs a GPT model as the brain, stores memories in a timeline, keeps a private vault, and can export your data. It adapts tone based on sentiment and can roleplay as you when asked.")
    st.markdown("**Notes & safety:**")
    st.markdown("- Vault uses strong encryption (Fernet) when 'cryptography' is available; otherwise a demo fallback is used — do not store real secrets without verifying encryption availability.")
    st.markdown("- Your OpenAI API key must be configured in Streamlit Secrets as `OPENAI_API_KEY`.")
    st.markdown("- This is still a prototype. For stronger privacy guarantees, add server-side encryption and a secure key manager.")

# End of file
