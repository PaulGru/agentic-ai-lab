# Agentic AI Lab — V8.5.1: Verified Results Ingestion

## Intention pédagogique

Nous construisons les briques V8 d'un scientifique IA autonome. **Nous ne
menons pas encore notre recherche sur la robustesse des LLM** et n'utilisons
pas Jean Zay. Le pilote V8.4.6 produit deux moyennes arithmétiques calculées
dans Docker ; ce sont des *tests de l'infrastructure*, pas des observations
scientifiques permettant d'inférer un effet causal.

### Le rôle de la brique

- **V8.4.6** prépare / exécute les jobs, vérifie les métadonnées et les
  artefacts, et persiste les reçus dans SQLite.
- **V8.5.1** relit le registre, reconstruit un rapport vérifié, normalise les
  métriques et produit une évaluation de qualité.
- **V8.5.2** (à venir) calculera les statistiques descriptives de manière
  déterministe sur un contrat de mesures plus général.

```text
ExperimentJob -> Docker -> DurableJobManager (SQLite)
                 |                 |
                 +--- artifacts --+
                                   |
                          build_report V8.4.6
                       (empreintes + métadonnées)
                                   |
                        ingest_verified_pilot
                                   |
                           AnalysisDataset
                                   +--> observations.json
                                   +--> quality.json
```

**Frontière de confiance :** la commande `ingest` exige `--protocol` et la base
SQLite des jobs, et appelle `build_report`. Elle n'accepte pas un rapport JSON
isolé comme preuve de provenance. L'empreinte SHA-256 sert à détecter une
modification accidentelle ou incohérente ; ce n'est PAS une signature
cryptographique qui authentifie l'auteur de données.

## Prérequis

- Protocole `examples/v846_protocol.json` inchangé depuis la campagne approuvée.
- Les deux jobs V8.4.6 ont été **approuvés et exécutés**, et leurs reçus
  sont présents dans `runs_control/v846_jobs.sqlite3`.
- Dépendances Python existantes (`pydantic`), sans Docker actif, PostgreSQL,
  clé API, ni accès HPC pour cette étape de lecture et d'analyse.

Si V8.4.6 n'a pas encore exécuté la campagne, lancer auparavant
`python -m scripts.demo_v846 ...` selon la documentation précédente. **Ne pas
rejouer un job existant sous le même identifiant.**

## Commandes PowerShell

```powershell
python -m unittest discover -s tests -p "test_v851_ingestion.py" -v

python -m scripts.demo_v851 ingest `
    --protocol examples/v846_protocol.json `
    --jobs-db runs_control/v846_jobs.sqlite3

python -m scripts.demo_v851 inspect `
    --out-dir runs_control/v851_analysis/v846-pilot-001
```

Les fichiers `observations.json` et `quality.json` sont créés dans
`runs_control/v851_analysis/v846-pilot-001/`. Si un fichier existe déjà avec
un contenu divergent, le programme refuse de l'écraser. Un export identique
peut être réexécuté.

## Comprendre les champs

- `run_id` : identifiant d'une exécution du pilote (baseline / variant).
- `value` : la **moyenne** mesurée par ce job ; elle doit valoir 3.0 / 6.0
  pour le protocole fourni et une exécution correcte.
- `input_value_count` : nombre d'entiers entrés dans le calcul ; **n'est pas**
  le nombre de répétitions statistiques indépendantes.
- `job_sha256` : empreinte du `ExperimentJob` validé.
- `artifacts_sha256` : empreinte des fichiers persistés.
- `source_report_sha256` : empreinte du rapport V8.4.6 vérifié et reconstruit.
- `simulated` : si `true`, un transport de test a été utilisé.
- `scientific_evidence: false` et `ready_for_scientific_inference: false` :
  aucune conclusion scientifique ni p-valeur ne découle de ce pilote.

**Inspect vs ingest :** `inspect` vérifie seulement l'intégrité *mutuelle*
des JSON exportés (empreinte du dataset, comptes), pas leur authenticité
face à SQLite. Pour une nouvelle vérification de la provenance, relancer
`ingest` avec le protocole et la base SQLite d'origine.

## Tests de sécurité et de validité

Les tests utilisent un transport explicitement marqué simulé, sans Docker :
refus de jobs incomplets, d'un protocole modifié, d'une empreinte d'artefact
corrompue, de valeurs non finies, de doublons, d'un export divergeant, et
de l'utilisation du pilote comme preuve de recherche. Ils testent aussi la
reconstruction après réouverture de SQLite.

## Limites et suite

V8.5.1 ne fait **aucun** test de significativité, intervalle de confiance,
régression, comparaison entre conditions ou interprétation scientifique. La
structure `AnalysisDataset` est une première normalisation, mais l'adaptateur
`ingest_verified_pilot` est propre à `synthetic.mean` ; nous ajouterons
plus tard un registre d'adaptateurs pour différentes métriques scientifiques.

Le statut `complete_for_pilot` signifie que tous les jobs du pilote ont des
résultats, **pas** qu'une étude est suffisamment répliquée. Dans notre future
étude réelle, il faudra documenter les tâches, seeds, répétitions, conditions,
plan expérimental, contrôles de multiplicité et règles d'exclusion AVANT
de calculer des conclusions.
