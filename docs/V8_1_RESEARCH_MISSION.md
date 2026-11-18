# V8.1 — Research Mission & Planning

## But

Convertir `research_brief.md` en une proposition de mission structurée et revue humainement. Aucun code expérimental, source bibliographique ou manuscrit n'est produit ici.

## Architecture

`research_brief.md` → `parse_brief` (validation des sections) → `MissionPlanner` (AgentRunner + provider actuel, recherche web désactivée) → `ResearchMission` (Pydantic) → `MissionWorkflow` (LangGraph + checkpoint) → `interrupt` (approbation humaine) → `missions/<thread>/mission.json`.

## Contrat d'entrée

En-têtes Markdown obligatoires : `## Question`, `## Contexte`, `## Périmètre`, `## Contraintes`, `## Critères de réussite`.

Le fichier doit contenir entre 100 et 12 000 caractères. `## Hypothèses initiales` est facultatif. La validation syntaxique ne garantit pas la qualité scientifique.

## Démarrage sous Windows / PowerShell

Depuis la racine du dépôt, avec les dépendances Python et le brief local :

```powershell
docker compose -f compose.yaml -f compose.v74.yaml up -d --build db
python -m scripts.demo_v81 start --thread robustness-001 --brief research_brief.md
python -m scripts.demo_v81 status --thread robustness-001
python -m scripts.demo_v81 resume --thread robustness-001 --decision approve
```

Répondre `OUI` pour approuver. L'artefact `missions/robustness-001/mission.json` est alors créé (sans écrasement).

Pour refuser :

```powershell
python -m scripts.demo_v81 resume --thread robustness-001 --decision reject
```

Pour une nouvelle proposition après rejet, choisir un nouvel identifiant de thread. Ne pas supprimer les checkpoints historiques.

## Tests

```powershell
python -m unittest tests.test_v81_mission -v
python -m unittest discover -s tests -v
```

Si les dépendances sont installées seulement dans Docker, les exécuter via `docker compose run --rm --entrypoint python lab -m unittest discover -s /app/tests -v` après reconstruction de l'image. La démonstration CLI nécessite le port local PostgreSQL de `compose.v74.yaml`.

## Points de vigilance

- Les empreintes SHA-256 lient l'approbation au brief et au contenu exact du plan. Elles ne prouvent pas sa qualité ni ne remplacent une authentification.
- Le modèle peut proposer des hypothèses contestables : revoir le plan avant validation.
- Aucun résultat, citation ou expérience n'est créé pendant V8.1.
- Les checkpoints sont dans le namespace `v81-mission:` : ne pas réutiliser les identifiants V7.4.
- L'export approuvé est volontairement séparé du checkpoint pour les versions suivantes.
- La reprise avec `status` et `resume` ne doit pas relancer le LLM.

## Définition de terminé

Tests verts ; invalid brief rejeté avant appel LLM ; plan structuré ; interruption HITL ; refus effectif ; approbation liée au bon digest ; redémarrage possible via PostgreSQL ; export exploitable par V8.2.
