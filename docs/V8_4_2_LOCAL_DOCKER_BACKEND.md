# V8.4.2 — Local Docker Execution Backend

## Pourquoi cette brique ?

La V8.4.1 définissait un **contrat** (`ExperimentJob`) et un backend simulé.
La V8.4.2 ajoute un backend qui **exécute réellement** un workload Python
stdlib connu dans le sandbox Docker de V7, sans dépendre des LLM scientifiques
(OpenAI/Anthropic) ni de Jean Zay.

Les LLM qui orchestrent la recherche appartiennent à `providers/` ; les
modèles/algorithmes étudiés sont des workloads du moteur `execution/`.

## Fichiers ajoutés

- `execution/workloads.py`: registre **allowlist** V8.4.2 (un seul workload
  `synthetic.mean`), validation des paramètres, code source statique et
  vérification de l'enveloppe de sortie JSON.
- `execution/backends/docker_transport.py`: encapsulation de Docker `run`
  et retrait du conteneur, avec limites de ressources, timeout, annulation,
  sorties bornées et protections héritées de V7.
- `execution/backends/local.py`: implémente `ExecutionBackend` (`submit`,
  `status`, `cancel`, `collect`) et `wait` pour la CLI.
- `scripts/tutorials/local_docker.py`: `check` sans effet, puis `run` avec confirmation
  humaine explicite.
- `tests/test_v842_local.py`: tests offline, avec **transport simulé** ;
  ils ne prétendent pas valider Docker réellement.
- `runs_local/.gitignore`: exclut du dépôt les artefacts locaux.

## Contrat inchangé

Utilise exactement `examples/v841_job.json` et `ExperimentJob` version
`8.4.1`. Aucun changement aux modules ou aux checkpoints précédents.

Le workload autorisé `synthetic.mean` calcule réellement `count`, `mean`,
`minimum` et `maximum` sur des valeurs numériques bornées. Il produit deux
objets JSON encapsulés dans stdout, sans montage de volume :

- `metrics.json`: mesures numériques effectives du workload.
- `run_metadata.json`: identifiants du job et de son empreinte, seed, version
  Python et provenance `docker_sandbox`.

La CLI conserve ces objets sur l'hôte seulement **après** validation du
message renvoyé par Docker. Elle crée également `receipt.json`.

## Étapes Windows / PowerShell

À la racine de `agentic-ai-lab`, sur une branche dédiée :

```powershell
python -m unittest discover -s tests -p "test_v842_local.py" -v
python -m unittest discover -s tests -v
python -m scripts.tutorials.local_docker check --job examples/v841_job.json
```

Vérifier que Docker Desktop est démarré, et construire l'image **déjà
utilisée en V7** (aucun package ou accès réseau pendant le job lui-même) :

```powershell
docker build -t agentic-lab-sandbox:v7.2 -f sandbox/Dockerfile sandbox
docker image inspect agentic-lab-sandbox:v7.2
```

Exécuter un vrai workload :

```powershell
python -m scripts.tutorials.local_docker run --job examples/v841_job.json
```

Le programme montre le job et son SHA-256, puis demande de taper `EXECUTE`.
Après réussite :

```text
runs_local/synthetic-mean-001/
    metrics.json
    run_metadata.json
    receipt.json
```

La moyenne attendue pour `[1, 2, 3, 4, 5]` est `3.0`. Les valeurs ne sont
pas une preuve de recherche (uniquement un exercice de validation technique).

Pour une **nouvelle** exécution, utiliser un autre `job_id` et donc un autre
chemin de sortie. Le programme n'écrase jamais les résultats existants.

## Limites et sécurité

- **Autorisation** : `LocalDockerBackend()` refuse les soumissions par
  défaut. La CLI fait confirmer l'exécution avant de créer un backend
  `execution_authorized=True`. Il ne s'agit pas encore d'une approbation
  LangGraph persistante (prévue en V8.4.5).
- **Allowlists** : seul `synthetic.mean` est accepté. Aucun programme, script,
  commande shell, URL, paquet ou modèle LLM dynamique n'est accepté dans le
  document JSON. Les paramètres sont traités comme **données**.
- **Isolation** : réseau interdit, conteneur en lecture seule, pas de volume
  hôte, utilisateur non privilégié, capacités supprimées, seccomp Docker,
  pas de téléchargement d'image à l'exécution.
- **Ressources** : jusqu'à 2 CPU, 512 MB, 0 GPU et 2 minutes. Les limites
  déclarées par `ExperimentJob` sont effectivement passées à Docker et
  vérifiées par la politique locale. La limite de temps est gérée par le
  processus hôte, qui force l'arrêt du conteneur ; elle n'est pas une
  réservation Slurm.
- **Sorties** : limitées à 8192 octets au total, artefacts avec noms fixes,
  empreinte de contrat et valeurs numériques vérifiées.
- **Durabilité** : état et contrôle d'idempotence **en mémoire seulement**.
  Un redémarrage pendant un job peut faire perdre le lien avec son conteneur ;
  les mécanismes de reprise, reçus durables, réconciliation et gestion des
  orphelins sont l'objet de V8.4.4. Ne pas automatiser des jobs coûteux
  avec cette version.
- **Reproductibilité** : l'image est encore référencée par un **tag Docker**,
  pas par digest immuable ; nous collecterons la version exacte de l'image
  dans une future étape. Le code du workload est statique et la seed est
  préservée dans les métadonnées, mais ne joue pas de rôle pour une moyenne
  déterministe.
- **Tests** : l'unittest simule le transport Docker. Une preuve d'exécution
  réelle doit venir de la commande `run`, avec Docker fonctionnel.

## Compatibilité et prochaines étapes

Cette brique n'utilise ni `v84_bindings.json` (ancien canary inadéquat), ni
une connexion à Jean Zay, ni une expérience sur la robustesse des LLM. Nous
ne lançons pas encore une étude scientifique de grande échelle.

V8.4.3 ajoutera un transport Slurm **simulé** pendant le développement, puis
V8.4.4 la persistance et la surveillance des jobs. La validation sur Jean
Zay restera une étape distincte une fois l'accès obtenu.
