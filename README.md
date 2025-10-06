# EchoSoul — adaptive, memoryful AI companion (demo)

This repository contains a Streamlit-based scaffold for **EchoSoul**, a personal AI companion with persistent memory, encrypted vault, adaptive personality, emotion detection, and a simulated live-call demo.

## Features included in this scaffold
- Persistent memory stored in SQLite (`echosoul.db`).
- Encrypted Memory Vault using `cryptography.Fernet`. Set `ECHOSOUL_KEY` in your environment.
- Simple text-based emotion detection (TextBlob) as a placeholder.
- Adaptive persona & tone mechanics are stored in profile and used when generating responses.
- Time-Shifted Self simulator (simple persona modifier).
- Simulated Live Call capture via `streamlit-webrtc` (requires `ffmpeg` and additional OS packages).
- Placeholder where you can plug an LLM (OpenAI, local model) and voice providers (Twilio / Agora / WebRTC).

## Deploying to Render (recommended with Docker)
1. Create a Render Web Service and connect your GitHub repo.
2. Set environment variables in Render:
   - `ECHOSOUL_KEY`: Fernet key (generate with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`)
   - Optionally `OPENAI_API_KEY` for LLM integration.
   - `PORT` is set by Render automatically; the container uses it.
3. Deploy using the provided `Dockerfile`.

## Local development
1. Create a venv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
