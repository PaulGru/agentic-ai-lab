
# Agentic AI Lab

Projet pédagogique pour apprendre à construire des
systèmes agentiques avec Python, des LLM et LangGraph.

## Objectif

Construire progressivement un assistant capable de :

- Rechercher des informations sur Internet.
- Effectuer des calculs mathématiques.
- Orchestrer plusieurs agents spécialisés.
- Exécuter et analyser des expériences Python (V7).

## Architecture actuelle

- Python : logique applicative
- OpenAI / Anthropic : fournisseurs LLM
- LangGraph : orchestration et checkpoints
- FastAPI : API HTTP
- PostgreSQL : persistance
- Docker Compose : environnement local

L'architecture conserve une abstraction multi-provider.

## Fonctionnalités

- Appels d'outils Python
- Routage et handoffs entre agents
- Human-in-the-loop (HITL)
- Checkpoints persistants
- Recherche Web native
- Audit de métadonnées de sources
- Pipeline de recherche multi-agent

## Démarrage local

Créer un fichier `.env` contenant les variables
nécessaires à Docker Compose et aux fournisseurs LLM.

Ne jamais committer les clés API.

Démarrer :

    docker compose up --build --wait -d api

Vérifier :

    http://127.0.0.1:8000/health

## API

- POST /workflows
- GET /workflows/{thread_id}
- POST /workflows/{thread_id}/decision
- POST /research
- GET /research/{thread_id}
- POST /research/pipeline
- GET /research/pipeline/{thread_id}

Les endpoints applicatifs sont protégés par X-API-Key.
Le endpoint /health est public.

## Tests

Tests Python :

    docker compose run --rm --entrypoint python lab -m unittest discover -s /app/tests -v

Tests HTTP :

    docker compose exec -T api python /app/scripts/smoke_v54.py

## Roadmap

- V0–V4 : fondations agentiques et orchestration
- V5 : Docker, API, persistance et sécurité
- V6 : recherche et consolidation
- V7 : agents mathématiques et expérimentaux

Le RAG est volontairement reporté.

## Déploiement

Exécution locale avec Docker Compose.

Le déploiement cloud n'est pas nécessaire pour
le développement actuel.
