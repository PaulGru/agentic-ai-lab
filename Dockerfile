FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du projet
COPY . .

# Utilisateur non privilégié et dossier persistant
RUN useradd --system --uid 10001 lab \
    && mkdir -p /app/data \
    && chown lab:lab /app/data

USER lab

# Les chemins relatifs de la V4.5 partiront d'ici
WORKDIR /app/data

ENTRYPOINT ["python", "/app/demo_v45.py"]