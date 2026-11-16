
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Exécution avec un utilisateur non privilégié
USER 10001:10001

WORKDIR /tmp

# Lit le programme depuis stdin.
# N'exécute rien tant qu'aucun code n'est fourni.
ENTRYPOINT ["python", "-I", "-B", "-"]
