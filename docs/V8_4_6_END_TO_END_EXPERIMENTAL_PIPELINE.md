# Agentic AI Lab — V8.4.6: End-to-End Experimental Pipeline

## Intention pédagogique

**V8 = construction des briques ; pas encore de recherche sur la robustesse.**
Le démonstrateur fait exécuter deux moyennes synthétiques par Docker, récolte
les artefacts vérifiés et produit un rapport reproductible. Les LLM OpenAI /
Anthropic pilotent le *chercheur* dans les autres modules, mais n'apparaissent
**pas** dans les modèles étudiés ici : aucun appel API n'est effectué.

Cette V8.4.6 ne connecte pas Jean Zay. Le simulateur Slurm de V8.4.3 reste
un module indépendant ; le parcours intégré utilise le backend Docker **réel**.

## Architecture

```text
examples/v846_protocol.json  (Pilote synthétique + ressources)
         |
         v
execution/campaign.py        (Validation stricte + 2 ExperimentJob)
         |
         v
execution/campaign_workflow.py (LangGraph + HITL campagne entière)
         |                        (approbation exacte SHA-256)
         v
execution/job_manager.py     (intention SQLite avant effet externe)
         |
         v
execution/backends/local.py  (workers Docker, dans le processus resume)
         |
         v
execution/workloads.py       (synthetic.mean, programme autorisé)
         |
         v
execution/backends/docker_transport.py (isolation V8.4.2)
         |
         v
execution/job_manager.py     (collecte + vérification + SHA-256)
         |
         v
execution/campaign.py        (rapport déterministe, export JSON)
```

### Pourquoi un `PilotProtocol` plutôt qu'un `v84_bindings.json` ?

`PilotProtocol` décrit **des calculs à exécuter**, pas des modèles LLM qui
servent de chercheurs, ni des modèles testés sur Jean Zay. Il se transforme en
`ExperimentJob` génériques par une liste de workloads autorisés. À terme,
d'autres workloads pourront être enregistrés (après validation de leur
sécurité et reproductibilité) sans changer le contrat d'exécution.

## Prérequis

- Patches V8.4.1 à V8.4.5 appliqués.
- `pydantic`, `python-dotenv`, `langgraph`, `langgraph-checkpoint-postgres`,
  `psycopg` déjà présents dans l'environnement V8.4.5.
- Docker Desktop opérationnel et image `agentic-lab-sandbox:v7.2` créée en V8.4.2.
- PostgreSQL local via `compose.yaml` + `compose.v74.yaml` et `.env`
  contenant `POSTGRES_PASSWORD`.
- Aucune clé OpenAI/Anthropic nécessaire et aucun accès HPC.

Si l'image sandbox manque :

```powershell
docker build -t agentic-lab-sandbox:v7.2 -f sandbox/Dockerfile sandbox
```

Démarrer la base :

```powershell
docker compose -f compose.yaml -f compose.v74.yaml up -d db
```

## 1. Prévisualiser les jobs (aucun Docker, aucune base)

```powershell
python -m scripts.demo_v846 plan --protocol examples/v846_protocol.json
```

Deux jobs sont générés : `v846-pilot-001-baseline` et
`v846-pilot-001-variant`, avec spécifications et empreintes distinctes.

## 2. Préparer le workflow HITL (aucune soumission)

```powershell
python -m scripts.demo_v846 start `
  --thread v846-pilot-thread-001 `
  --protocol examples/v846_protocol.json

python -m scripts.demo_v846 status --thread v846-pilot-thread-001
```

Le graph s'interrompt sur `v846_campaign_approval`. **Aucun job Docker n'a
encore été lancé**. L'approbation porte sur l'empreinte du protocole et sur
les deux jobs affichés.

## 3. Approuver et exécuter (Docker réel)

```powershell
python -m scripts.demo_v846 resume `
  --thread v846-pilot-thread-001 `
  --decision approve
```

Vérifier les empreintes et entrer `APPROVE` lorsque le programme le demande.
La commande **reste active pendant l'exécution des deux jobs**, car les
workers V8.4.2 ne persistent pas au redémarrage d'un processus Python.

Pour tester le refus, créer un **nouvel identifiant de thread et un nouveau
`campaign_id`** (dans une copie du JSON), puis choisir `--decision reject`.
Aucune soumission ne doit se produire.

## 4. Examiner les sorties

```powershell
python -m scripts.demo_v846 status --thread v846-pilot-thread-001
python -m scripts.demo_v846 report --thread v846-pilot-thread-001
```

Le rapport se trouve sous `runs_control/v846_reports/v846-pilot-001.json`.
Il expose la provenance des deux jobs, les empreintes des artefacts, les
métriques et la différence entre la dernière et la première moyenne.

**Résultat attendu** (à vérifier par l'exécution réelle) :

```json
{
  "run_means": {"baseline": 3.0, "variant": 6.0},
  "last_minus_first_mean": 3.0,
  "number_of_runs": 2,
  "method": "deterministic comparison of synthetic arithmetic means"
}
```

Le rapport doit aussi indiquer `simulated: false`,
`scientific_evidence: false`. Le premier dit que les calculs ont réellement
été exécutés dans Docker ; le second rappelle que ces deux moyennes ne
constituent pas une étude scientifique.

La commande `report` peut reconstruire le fichier à partir du registre SQLite
après un redémarrage, à condition que les deux jobs aient déjà été collectés.
Elle refuse d'écraser un fichier de rapport divergent.

## 5. Tests

```powershell
python -m unittest discover -s tests -p "test_v846_campaign.py" -v
python -m unittest discover -s tests -p "test_v846_workflow.py" -v
python -m unittest discover -s tests -v
```

Le premier ensemble vérifie sans Docker ni PostgreSQL : validation de
protocole, transformation en jobs, collecte, intégrité, idempotence du rapport
et reconstruction depuis SQLite.

Le second (requiert `langgraph`, avec `InMemorySaver`) vérifie : HITL avant
soumission, rejet, hash invalide, backend invalide et succès complet avec
transport simulé **explicitement marqué simulated=true**. Ce mock n'est **pas**
un test d'exécution Docker. La validation réelle passe par la CLI.

## Limites importantes

1. **Processus local :** les workers Docker V8.4.2 sont en mémoire. Si
   `resume` est interrompu brutalement, le registre peut devoir passer en
   `needs_reconciliation`. Ne pas relancer sous le même ID à l'aveugle. Examiner
   `docker ps -a` et le registre avant toute nouvelle exécution.
2. **Pas de scheduler HPC réel :** les scripts Slurm simulés ne garantissent
   pas la compatibilité avec Jean Zay. L'adaptateur Slurm distant sera
   validé après attribution de l'accès.
3. **Pas de code arbitraire :** `synthetic.mean` reste le seul workload
   disponible pour le démonstrateur, uniquement en environnement Docker isolé.
4. **Pas d'analyse scientifique :** moyenne et différence ne sont que des
   contrôles numériques, sans inférence statistique, dataset scientifique,
   facteur Dense/MoE, modèle de recherche ou article ACL.
5. **Pas d'agent autonome intégré au-dessus :** ce pipeline démontre le
   contrat d'exécution complet ; son branchement au Research Planner et à
   l'analyse statistique véritable relève des prochaines briques V8.

## Critères de réussite

- [ ] `plan` produit deux jobs cohérents, sans appel modèle.
- [ ] `start` s'interrompt avant toute soumission.
- [ ] `resume` requiert une approbation explicite.
- [ ] Deux conteneurs Docker exécutent un calcul réel sous restrictions.
- [ ] Deux artefacts par job sont validés et conservés par le ledger.
- [ ] `report` fonctionne dans un nouveau processus Python.
- [ ] Après relance, aucun job identique n'est soumis une seconde fois.
- [ ] Le statut et les sorties indiquent clairement `simulated: false`.

## Suite

**V8.5 — Results & Statistical Analysis (engineering)** : formaliser l'API
entre le moteur d'exécution et l'analyse, enrichir la traçabilité, puis
implémenter des validations statistiques reproductibles sur des datasets
synthétiques. Notre question scientifique sur Jean Zay reste pour la V9.
