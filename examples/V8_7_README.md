# V8.7 — Démonstration de mémoire avec données fictives

Les missions, articles, extraits et scores utilisés ci-dessous sont les exemples
fictifs de V8.6 et V8.5. Ils ne démontrent aucune découverte scientifique.
Les sorties restent sous `runs_control/`, exclu de Git.

Depuis la racine du dépôt, dans son environnement Python :

```powershell
python -m scripts.demo_v87_memory --project demo init
python -m scripts.demo_v87_memory --project demo import-context --path examples/v86_design_context.json
python -m scripts.demo_v87_memory --project demo search --query "robustesse agents" --kind hypothesis
```

La recherche retourne des souvenirs avec leur statut et leurs références,
sans appel LLM. Refaire la dernière commande dans un nouveau terminal montre
que la récupération ne dépend pas du processus ou d'un `thread_id`.

Pour faire consulter cette mémoire par le critique, puis y conserver sa revue :

```powershell
python -m scripts.demo_v86 --memory-db runs_control/v87_memory.sqlite3 --project demo --store runs_control/v87_demo_reviews.sqlite3 offline --input examples/v86_review_input.json --out-dir runs_control/v87_demo_report
python -m scripts.demo_v87_memory --project demo search --query "indépendance réplication" --kind lesson
```

Sans draft, ce parcours produit une revue seule. Une nouvelle invocation avec
le même identifiant est refusée : copier `v86_review_input.json` et changer
`review_id` pour un autre travail. Les versions précédentes sont conservées.

Pour PostgreSQL, utiliser `--postgres` au lieu du fichier SQLite et configurer
`RESEARCH_MEMORY_DATABASE_URL` dans l'environnement local. Voir le guide V8.7
avant toute connexion ou initialisation d'un service partagé.
