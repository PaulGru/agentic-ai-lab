# V8.2 — Scientific Literature Review

## Objectif pédagogique

Partir de `missions/<thread>/mission.json` **approuvé en V8.1** pour effectuer une découverte bibliographique réelle, puis une synthèse traçable à partir des résumés retrouvés. Ne confondez pas une notice OpenAlex avec une vérification du texte intégral.

## Pipeline

```
mission.json (V8.1, empreinte vérifiée)
  -> discover : OpenAlex, max 4 requêtes × 5 notices (configurable, max 10 par requête)
  -> analyze  : AgentRunner + provider actuel, sans outils, sans recherche web native
  -> approval : interrupt LangGraph, empreintes mission/catalogue/revue
  -> export   : literature.json + literature.md
```

Namespace de checkpoints : `v82-literature:<thread>` ; préserve tous les anciens namespaces.

## Pourquoi OpenAlex ?

Le service fournit des *métadonnées* bibliographiques, DOI et parfois un résumé sous forme d'index inversé. Notre code reconstruit le résumé quand il existe. Une citation ne peut être validée que si le `work_id` appartient au corpus et si un extrait de 25 mots maximum est présent dans le résumé **après normalisation de la casse et des espaces**. Cela ne valide pas l'interprétation sémantique faite par le LLM.

Requêtes limitées à la recherche *titre + résumé* : `search.title_and_abstract`. Le classement est celui renvoyé par OpenAlex ; il n'existe pas encore de sélection systématique avec critères d'inclusion/exclusion. Les doublons DOI / OpenAlex sont retirés. Pas de navigation vers des sites arbitraires, pas d'exécution de code généré par le LLM.

L'API reste accessible à faible échelle sans clé ; pour augmenter le quota, définissez `OPENALEX_API_KEY` dans `.env` (non suivi par Git). En cas de 429, réduire le rythme / utiliser une clé ; le code échoue sans produire une revue trompeuse.

## Utilisation (PowerShell, à la racine du dépôt)

Prérequis : patch V8.1 installé, mission approuvée et exportée, Python avec les dépendances du dépôt, PostgreSQL lancé sur le port 55432 via `compose.v74.yaml`, clés LLM dans `.env`.

```powershell
docker compose -f compose.yaml -f compose.v74.yaml up -d db
python -m scripts.demo_v82 start --thread robustness-001 --mission missions/robustness-001/mission.json
python -m scripts.demo_v82 status --thread robustness-001
python -m scripts.demo_v82 resume --thread robustness-001 --decision approve
# Si besoin, ré-exporter sans relancer le LLM :
python -m scripts.demo_v82 export --thread robustness-001
```

Changer `--thread` pour recommencer une nouvelle revue ; un thread existant ne peut être réécrit. `--max-queries` de 1 à 4 et `--per-query` de 1 à 10 peuvent être précisés à `start` pour réduire les appels.

## Tests hors ligne

```powershell
python -m unittest discover -s tests -p "test_v82_literature.py" -v
python -m unittest discover -s tests -v
```

Les tests simulent OpenAlex et le LLM ; aucun coût API, aucune connexion réseau ni PostgreSQL ne sont nécessaires en test. Pour tester dans Docker après `--build`, utiliser `docker compose run --rm --entrypoint python lab -m unittest discover -s /app/tests -v`.

## Limites et futur

- La validation d'un extrait exact limite les citations inventées, mais ne prouve pas que le résumé soutient la phrase interprétée.
- Un digest SHA-256 détecte les modifications d'artefacts ; il **n'est pas** une signature d'identité ou une authentification.
- Une petite recherche bornée n'est **pas** une revue systématique. Les lacunes proposées sont des hypothèses, pas une preuve d'absence d'études.
- Les notices sans résumé sont conservées dans le catalogue, mais elles ne peuvent pas justifier les `findings`.
- Il faudra ensuite consulter les textes complets autorisés, tracer les critères d'inclusion et vérifier les protocoles des articles les plus importants avant toute conclusion ACL.
- La revue approuvée reste un document de travail, non une preuve de nouveauté ni un manuscrit publiable.

## Définition de terminé

Entrée V8.1 validée avant réseau, récupération OpenAlex réelle et bornée, catalogage avec DOI/ID et trace des requêtes, citations de résumés réellement présents, approbation HITL persistante, refus et mauvais digests bloqués, exports sans écrasement silencieux, tests historiques et V8.2 verts.
