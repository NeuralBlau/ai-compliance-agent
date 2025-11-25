# Startpunkt: Ein Python-Image mit allen notwendigen Tools
FROM python:3.11-slim

# Umgebungsvariablen für Tesseract (OCR)
ENV DEBIAN_FRONTEND=noninteractive

# Installiere Tesseract, Git und andere notwendige Abhängigkeiten
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    git \
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installiere AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm awscliv2.zip && \
    rm -rf aws

# Setze das Arbeitsverzeichnis
WORKDIR /app

# Kopiere die Abhängigkeiten und installiere sie
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Port für FastAPI-Server
EXPOSE 8000

# Optional: Befehl, der beim Start ausgeführt wird (z.B. für Entwicklung)
CMD ["sleep", "infinity"]
