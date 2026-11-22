# Commandes du laboratoire

Exécuter les commandes Python depuis la racine du dépôt avec `python -m`.

| Commande | Usage |
|---|---|
| `python -m scripts.research_workflow --help` | Piloter une mission depuis un brief. Les sous-commandes `progress` et `dashboard` affichent l'avancement. [Guide](../docs/WORKFLOW_RECHERCHE.md) |
| `python -m scripts.research_memory --help` | Consulter, importer et administrer la mémoire de recherche. [Guide](../docs/V8_7_RESEARCH_MEMORY.md) |
| `python -m scripts.experiment_workflow --help` | Piloter et suivre une expérience Python isolée avec approbation. [Sécurité du sandbox](../docs/V7_SECURITY.md) |
| `python -m scripts.pilots.tool_retry --help` | Préparer et suivre le pilote apparié de relance d'outil. [Guide du pilote](../docs/V9_PILOTE_RELANCE_OUTIL.md) |
| `python -m scripts.pilots.tool_retry_manuscript --help` | Reconstruire le brouillon du pilote depuis son journal. [Guide du pilote](../docs/V9_PILOTE_RELANCE_OUTIL.md) |

`scripts/tutorials/` contient les commandes pédagogiques pour reproduire une
étape isolée. Le tutoriel [de bout en bout sur données synthétiques](../docs/V8_9_RESEARCH_WORKFLOW.md)
donne un point de départ pour parcourir les étapes liées.

`scripts/checks/` contient les vérifications d'intégration de la CI.
`api_http` et `research_http` ciblent l'API démarrée. `critic_postgres`,
`memory_postgres` et `synthetic_workflow_postgres` vérifient la reprise
ou la persistance avec un PostgreSQL configuré pour les tests.
