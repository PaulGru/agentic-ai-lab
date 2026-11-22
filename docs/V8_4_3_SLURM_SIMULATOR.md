# Agentic AI Lab — V8.4.3: Slurm Preview & Offline Simulator

## Pourquoi

Slurm est un ordonnanceur HPC. Il dispose d'opérations comme `sbatch` (soumettre),
`squeue` (travail en cours), `sacct` (historique) et `scancel` (annuler).
Une même interface `ExecutionBackend` peut, à terme, faire fonctionner un
backend local et un adaptateur HPC. **Cette version NE communique PAS avec un
cluster Slurm** : c'est un contrat de soumission + un simulateur persistant.

## Fichiers

- `execution/backends/slurm_sim.py` : adaptation des états Slurm, aperçu du script `sbatch`, simulation.
- `scripts/tutorials/slurm_simulator.py` : affichage du script et cycle de vie fictif.
- `tests/test_v843_slurm_sim.py` : tests sans réseau.

## Commandes

Depuis la racine du dépôt (PowerShell) :

```powershell
python -m unittest discover -s tests -p "test_v843_slurm_sim.py" -v
python -m scripts.tutorials.slurm_simulator preview --job examples/v841_job.json
python -m scripts.tutorials.slurm_simulator simulate --job examples/v841_job.json
```

Le script preview ne contient que le workload autorisé `synthetic.mean`,
construit à partir d'un template de confiance. Aucun shell ni code fourni par
le LLM n'est exécuté. Le rendu est un prototype, **pas** un script de
production prêt à soumettre à Jean Zay (environnement et sorties à adapter).

Le simulateur accepte seulement `submitted -> running -> succeeded` ou
l'annulation, dans une base SQLite. La collecte retourne toujours
`simulated=true` et `files={}` : **aucune mesure scientifique réelle**.

## Documentation Slurm

- https://slurm.schedmd.com/sbatch.html
- https://slurm.schedmd.com/squeue.html
- https://slurm.schedmd.com/sacct.html
- https://slurm.schedmd.com/job_state_codes.html

L'adaptateur réel, après attribution d'un accès HPC, devra prendre en charge
l'authentification, le staging, le transfert d'artefacts, les allocations,
les partitions, les modules et les politiques locales du centre.
