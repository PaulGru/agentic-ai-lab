# V8.2 — Correctif HTTP 429 OpenAlex

## Pourquoi

OpenAlex peut renvoyer 429 en cas de quota quotidien épuisé, de dépassement de débit, ou de rythme limité sur un compte nouveau. Notre version précédente levait un `RuntimeError` sans consulter les en-têtes.

## Comportement

- Le client lit `X-RateLimit-Remaining`, `X-RateLimit-Reset` et `Retry-After`.
- Si `Remaining` vaut 0, le client arrête immédiatement.
- Si `Retry-After` est supérieur à 5 secondes, le client arrête immédiatement et affiche la recommandation.
- Sinon, il effectue au plus 2 nouvelles tentatives, avec un délai borné (1 puis 2 secondes au minimum).
- Aucun appel LLM ne s'effectue avant que la découverte soit complète.
- Les credentials restent dans le fichier `.env` ; le client transmet la clé via `Authorization: Bearer` (déjà pris en charge).
- Les tests utilisent des faux appels HTTP, sans connexion externe.

## Pour configurer le quota gratuit

Créer un compte OpenAlex, vérifier son email, copier la clé depuis **Settings → API key**. Dans `.env` (ignoré par Git) :

```dotenv
OPENALEX_API_KEY=ma_cle_personnelle
```

La clé ne doit pas être ajoutée au dépôt, aux traces ni aux tickets de bugs.

## Tester

```powershell
python -m unittest discover -s tests -p "test_v82_openalex_rate_limit.py" -v
python -m unittest discover -s tests -p "test_v82_literature.py" -v
```

L'échec 429 peut laisser un état LangGraph partiel : relancer avec un identifiant `--thread` différent, sans effacer les checkpoints PostgreSQL.

Documentation : https://help.openalex.org/api/authentication/
