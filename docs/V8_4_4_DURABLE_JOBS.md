# Agentic AI Lab — V8.4.4: Durable Job Management

## Pourquoi

Un checkpoint LangGraph enregistre l'état d'un workflow, mais **ne garantit
pas** qu'une commande externe (`sbatch`, exécution Docker, paiement) soit
exécutée exactement une fois. Un crash entre une soumission externe et
l'enregistrement du résultat rend le statut ambigu.

## Principe

`DurableJobManager` utilise SQLite, indépendamment du checkpointer LangGraph :

1. Valider le job et calculer une empreinte stable.
2. Inscrire de manière transactionnelle une intention `submission_reserved`.
3. Appeler une fois `backend.submit(job)`.
4. Si la réponse est fiable, stocker le `JobHandle` et l'état `submitted`.
5. Si l'issue est ambiguë, stocker `needs_reconciliation` **sans relancer**.
6. Rafraîchir le statut à la demande; collecter les artefacts uniquement après succès.
7. Stocker l'enveloppe d'artefacts et son SHA-256 pour détection de corruption.

La clé logique est `job_id + spec_sha256 + backend`. Le même `job_id`
avec une autre spécification ou un autre backend est refusé.

Les états `needs_reconciliation` nécessitent une intervention de l'opérateur :
identifier l'exécution côté backend, vérifier et, à terme, rattacher son
handle après vérification. Il n'existe **aucun** retry automatique.

## Tests et démonstration (sans HPC)

```powershell
python -m unittest discover -s tests -p "test_v844_job_manager.py" -v
python -m scripts.tutorials.durable_jobs submit --job examples/v841_job.json
python -m scripts.tutorials.durable_jobs status --job examples/v841_job.json
python -m scripts.tutorials.durable_jobs advance --job examples/v841_job.json
python -m scripts.tutorials.durable_jobs advance --job examples/v841_job.json
python -m scripts.tutorials.durable_jobs collect --job examples/v841_job.json
```

La dernière commande retourne des artefacts simulés vides. Le redémarrage entre les commandes
permet de valider la persistance. Utilise un autre `job_id` pour un nouveau scénario.

## Limites

- SQLite convient à un seul hôte de développement : ce n'est pas encore un service distribué.
- La preuve d'intégrité SHA-256 ne constitue ni une signature ni une preuve d'authenticité.
- Un handle local Docker V8.4.2 ne peut pas être récupéré après la disparition de
  son processus Python; le statut est alors classé ambigu, pas relancé.
- Le simulateur Slurm persiste dans sa propre base SQLite : elle doit être conservée avec
  le ledger pour un redémarrage cohérent.
- Ne supprime jamais les bases SQLite au milieu d'un test de reprise.
