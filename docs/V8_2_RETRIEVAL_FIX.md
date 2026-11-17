# V8.2 — Correctif de récupération de littérature

## Symptôme

`LiteratureCatalog.papers: List should have at least 3 items after validation, not 1`.

Ce n'est pas un problème de PostgreSQL. Une recherche bibliographique peut
retourner trop peu de résultats ou trop peu de résumés réellement disponibles.
La V8.2 conserve l'exigence d'au moins 3 notices distinctes et 2 résumés.

## Changements

- CLI: `--query` répétable pour remplacer **uniquement les termes de recherche**
  pendant la récupération; le `mission.json` approuvé n'est pas modifié.
- Diagnostic explicite pour chaque requête: total OpenAlex, résultats de la page,
  métadonnées rejetées, doublons, nouveaux articles acceptés et résumés utilisables.
- Arrêt propre avec message lisible si le corpus est encore insuffisant;
  aucun appel au LLM analyste n'est alors déclenché.
- Tests de non-régression et tests des requêtes de remplacement.

## Commandes PowerShell

Dans la racine du projet, **extraire le correctif en remplaçant** les quatre
fichiers existants (`research/`, `scripts/`, `tests/`) et ajouter cette documentation.

```powershell
python -m unittest discover -s tests -p "test_v82_literature.py" -v
python -m unittest discover -s tests -v

python -m scripts.demo_v82 start `
    --thread robustness-search-002 `
    --mission missions/robustness-001/mission.json `
    --per-query 10 `
    --query "LLM agents" `
    --query "agent robustness" `
    --query "tool use agents" `
    --query "multi agent systems"

python -m scripts.demo_v82 status --thread robustness-search-002
```

Le nouveau thread est nécessaire parce qu'un `start` antérieur peut avoir
laissé un checkpoint V8.2 partiel. Ne pas supprimer les volumes Docker ni
les checkpoints. Lorsque la revue est approuvée, les exports se retrouvent
dans `missions/robustness-search-002/`, correspondant au nouveau thread.

**Notes scientifiques:** des requêtes plus larges augmentent le rappel mais
peuvent réduire la pertinence. La validation des 3 notices et des 2 résumés
n'est pas une preuve de suffisance de la revue pour un article ACL. La personne
qui approuve doit contrôler la pertinence et la couverture, pas seulement le
nombre de résultats.
