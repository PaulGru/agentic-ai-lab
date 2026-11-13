# Agentic AI Lab — Architecture de référence (V6.7)

## Objectif

Construire un assistant multi-provider capable de rechercher des informations,
calculer, orchestrer plusieurs agents, puis expérimenter en Python en V7.

La V6.7 stabilise la structure, **sans ajouter d'exécution de code arbitraire**.
Le déploiement actuel reste local sous Docker Compose.

## Parcours HTTP

```text
Client HTTP
    |
    v
api/app.py (FastAPI)
    +-- api/core.py : X-API-Key, lifespan, PostgreSQL, Lock
    +-- api/routes/workflows.py : demo HITL historique
    |       -> workflows/legacy.py -> AgentToolGraph
    +-- api/routes/research.py
            +-- /research -> research/factory.py -> AgentToolGraph
            +-- /research/pipeline -> research/pipeline.py -> StateGraph
```

## Rôles et frontières

| Module | Responsabilité | Ne doit pas faire |
|---|---|---|
| `api/` | Validation HTTP, authentification, réponse | Logique des modèles ou exécution arbitraire |
| `providers/` | Adapter OpenAI / Anthropic vers `LLMResponse` | Gérer les routes HTTP |
| `agent/` | Exécuter le protocole LLM/outils | Choisir une infrastructure Docker |
| `orchestration/` | États LangGraph, handoffs, approbations | Dépendre de la recherche documentaire |
| `research/` | Recherche native, planification, synthèse, audit | Contrôler des actions sur le système hôte |
| `tools/` | Outils à entrée structurée et comportement explicite | Exécuter du code libre sans politique |
| `workflows/legacy.py` | Démonstration déterministe HITL V4.5/V5 | Prétendre être le futur orchestrateur général |
| `observability/` | Traces pédagogiques en mémoire | Être considéré comme un monitoring de production |

## États persistants

- Workflow historique : `thread_id` fourni par l'utilisateur.
- Recherche simple : `v6-research:{thread_id}`.
- Pipeline multi-agent : `v64-pipeline:{thread_id}`.

**Ne pas renommer ces préfixes ni les nœuds des graphes existants sans migration
et tests de reprise des checkpoints.** PostgreSQL stocke les checkpoints LangGraph.
Les volumes Docker doivent être conservés (`docker compose down -v` détruit
les volumes du projet).

## Authentification

La clé `AGENTIC_API_KEY` est exigée dans l'en-tête `X-API-Key` pour les routes
applicatives. `/health` est public. Le système utilise actuellement une clé
partagée et ne fournit pas d'isolation par utilisateur ou par projet.

Les secrets restent dans `.env` ou dans des configurations locales non suivies
par Git. Ils ne doivent pas être ajoutés aux traces ni aux réponses de l'API.

## Qualité des sources

`research/source_audit.py` valide les métadonnées et dédoublonne les URL ; cela
**ne vérifie ni le contenu des pages, ni les affirmations produites par le LLM**.
La recherche native chez les fournisseurs peut entraîner des coûts.

## Limites connues à préserver dans la roadmap

- Le service FastAPI est synchrone ; un verrou global protège les workflows,
  mais sérialise aussi les appels LLM potentiellement longs.
- `GET /research/...` construit actuellement le graphe avec un provider, même
  sans nouvelle génération ; cette dépendance sera à isoler lors d'une future
  amélioration de lecture des checkpoints.
- `workflows/legacy.py` utilise des providers simulés : c'est un démonstrateur
  de HITL et d'idempotence, pas un agent autonome en production.
- Le `Tracer` actuel est pédagogique. Pas encore de métriques de coût, durée,
  tokens et taux d'échec centralisées pour l'API.
- Les checkpoints seuls ne rendent pas les effets externes exactly-once.
  L'idempotence est la responsabilité des outils et des workflows.
- L'exécution de scripts générés par un LLM n'est **pas** autorisée dans
  le conteneur API actuel ; elle nécessite un sandbox distinct en V7.

## Contrôles avant de fusionner

```powershell
docker compose config --quiet
docker compose up --build --wait -d api
docker compose run --rm --entrypoint python lab -m unittest discover -s /app/tests -v
docker compose exec -T api python /app/scripts/smoke_v54.py
docker compose exec -T api python /app/scripts/smoke_v67.py
```

En CI, les tests n'appellent pas les véritables API OpenAI ou Anthropic.
Les parcours LLM et LangGraph sont simulés ou vérifiés avec de faux providers.
