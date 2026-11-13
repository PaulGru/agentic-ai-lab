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

# Working directory shared with the local test service
WORKDIR /app/data

# Default execution for the pedagogical test container
ENTRYPOINT ["python", "-m", "unittest", "discover", "-s", "/app/tests", "-v"]