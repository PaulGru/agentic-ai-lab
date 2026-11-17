# Agentic AI Lab — V8.4.5: LangGraph + HITL + Durable Execution

## Objectif

Faire fonctionner ensemble le contrat V8.4.1, le backend local V8.4.2,
le simulateur Slurm V8.4.3 et le ledger V8.4.4, sans LLM payant et sans HPC.
Le workflow est distinct des graphes historiques; ses checkpoints utilisent le
namespace `v845-execution:`.

```text
start -> validate -> approval (interrupt)
                      | approve + exact SHA-256 + backend
                      v
                     submit (ledger reserves before external call)
                      v
                     observe -> END
                      |
                      +-> status/advance outside graph: refresh ledger
```

L'approbation porte sur le job exact et sur le backend choisi. La simulation
Slurm ne crée aucun job sur un cluster. Le workflow ne produit ni article ni
résultats de recherche; il teste l'orchestration technique.

## Prérequis

- Branches V8.4.1 et V8.4.2 présentes dans le dépôt.
- `langgraph`, `langgraph-checkpoint-postgres`, `psycopg` dans l'environnement
  Python du projet (comme pour V7.4).
- `.env` avec `POSTGRES_PASSWORD` pour le démonstrateur Windows.
- PostgreSQL exposé sur `127.0.0.1:55432` via `compose.v74.yaml`.

```powershell
docker compose -f compose.yaml -f compose.v74.yaml up -d db
python -m unittest discover -s tests -p "test_v845_workflow.py" -v
```

## Exemple recommandé : Slurm simulé, avec approbation réelle

```powershell
python -m scripts.demo_v845 start --thread v845-demo-001 --job examples/v841_job.json
python -m scripts.demo_v845 status --thread v845-demo-001
python -m scripts.demo_v845 resume --thread v845-demo-001 --decision approve
python -m scripts.demo_v845 status --thread v845-demo-001
python -m scripts.demo_v845 advance --thread v845-demo-001
python -m scripts.demo_v845 advance --thread v845-demo-001
python -m scripts.demo_v845 status --thread v845-demo-001
```

La commande `resume` demande de saisir `APPROVE`. Pour tester le refus, utilise
un nouveau thread **et un nouveau job_id** (le `job_id` est un identifiant
unique dans le ledger), puis `--decision reject`.

Le workflow LangGraph se termine après une observation; les changements de
statut sont suivis dans le **ledger durable**, qui est autoritatif pour la
situation actuelle du job. Il est normal que le checkpoint LangGraph affiche
une ancienne valeur après `advance`.

## Option Docker local (réel, synthétique uniquement)

Avec Docker Desktop et l'image de V8.4.2 :

```powershell
python -m scripts.demo_v841 example --out examples/v845-local-job.json
```

Change `job_id` dans le nouveau fichier avant exécution; puis :

```powershell
python -m scripts.demo_v845 start --thread v845-local-001 --backend docker-local --job examples/v845-local-job.json
python -m scripts.demo_v845 resume --thread v845-local-001 --backend docker-local --decision approve
python -m scripts.demo_v845 status --thread v845-local-001 --backend docker-local
```

La commande `resume` reste active jusqu'à la fin du job local afin de ne pas
abandonner le thread Docker V8.4.2. Si le processus est brutalement interrompu,
**ne relance pas** le job sous le même ID. Réconcilie manuellement la présence
d'un conteneur orphelin; le ledger signalera un état incertain.

## Architecture et sécurité

- Pas de commandes arbitraires générées par LLM.
- Aucune connexion HPC à cette étape.
- Les jobs portent des empreintes, et le gestionnaire n'effectue pas de soumission
  automatique lorsque la réponse ou la reprise est ambiguë.
- Le login SSH, l'authentification, la politique de stockage et les allocations
  spécifiques à Jean Zay seront ajoutés **après** accès HPC.
- Seuls les travaux `synthetic.mean` déjà autorisés en V8.4.2 sont disponibles.
- Cette version n'est pas un ordonnanceur multi-utilisateur de production.

## Tests attendus

Les tests utilisent `InMemorySaver`, le simulateur SQLite et des objets simulés.
Ils vérifient l'interruption initiale, le refus, une empreinte invalide, un
backend incorrect, la reconstruction du graphe et l'absence de double soumission.
Ils ne requièrent pas PostgreSQL pour les tests unitaires, seulement pour la CLI.
