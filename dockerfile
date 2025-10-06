# Use an official Python runtime as a parent image
FROM python:3.10-slim

# set working dir
WORKDIR /app

# system deps for audio / webrtc
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      build-essential \
      && rm -rf /var/lib/apt/lists/*

# copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy app
COPY . /app

# set environment
ENV PYTHONUNBUFFERED=1

# expose port
EXPOSE 8080

# Run the Streamlit app. Render injects PORT in env.
CMD ["bash", "-lc", "streamlit run app.py --server.port ${PORT:-8080} --server.address 0.0.0.0"]
