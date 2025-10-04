# Dockerfile for EchoSoul (with spaCy model install)
FROM python:3.10-slim

WORKDIR /app

# Copy only requirements first for smarter caching
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libsndfile1 \
    libportaudio2 \
    libopus0 \
    libvpx7 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model (ensure model exists at build time)
RUN python -m spacy download en_core_web_sm

# Copy app files
COPY . .

EXPOSE 8501

# Use environment PORT when provided by Render / Railway
CMD ["streamlit", "run", "app.py", "--server.port=$PORT", "--server.address=0.0.0.0"]
