# Agentic AI Lab — V8.4 (étape 1) : premier moteur d'exécution réel

## Pourquoi une étape 1 ?

V8.3 produit un protocole pré-spécifié en texte structuré, mais ses niveaux
(`small`, `planner`, `retry`, etc.) ne correspondent pas encore à des modèles,
des commandes et des jeux de données exécutables. En déduire automatiquement
les implémentations serait arbitraire et invaliderait potentiellement le plan.

Cette première tranche de V8.4 ajoute un **canary d'ingénierie** avec de vrais
appels LLM, pour tester la chaîne technique, **sans présenter ces observations
comme des résultats scientifiques du plan V8.3**.

## Architecture

```text
experimental_design.json + design_matrix.json (V8.3 approuvés)
                |
                v
          verify_approved_design
                |
                v
    v84_bindings.json (définition humaine des niveaux)
                |
                v
   prepare_canary_plan (cellule / tâches / budget / empreinte)
                |
                v
    LangGraph prepare -> HITL approve -> execute
                                  |
                                  v
                receipt JSON réservé AVANT appel API
                                  |
                                  v
           OpenAI Responses API / Anthropic Messages API
                                  |
                                  v
              Score exact + tokens + latence + coût estimé
```

- `research/execution_contract.py` : intégrité V8.3, correspondance explicite
  des facteurs, pré-vérification du budget, template.
- `research/canary_runtime.py` : tâches synthétiques déterministes, simulation
  contrôlée d'une indisponibilité du calculateur local, appels LLM **réels**,
  scoring, reçu durable. Le calculateur est une opération arithmétique fixe,
  pas une exécution de code produit par le modèle.
- `research/execution_graph.py` : LangGraph, interruption HITL, espace de
  checkpoints `v84-canary:{thread}` (distinct de V7/V8.1/V8.2/V8.3).
- `scripts/demo_v84.py` : commandes Windows PowerShell.

## 1. Préparer la branche

Ne pars pas de `main` si V8.1-V8.3 n'y ont pas encore été fusionnées.
Sur une branche propre contenant V8.3 :

```powershell
git status
git switch -c feat/v8.4-execution-canary
```

Extrais le ZIP V8.4 à la racine du dépôt, sans remplacer les modules précédents.

## 2. Tests (aucune facturation)

```powershell
python -m unittest discover -s tests -p "test_v84_execution.py" -v
python -m unittest discover -s tests -v
```

Les tests V8.4 n'appellent ni OpenAI ni Anthropic ; ils utilisent un fake. Les
2 tests LangGraph sont actifs dans l'environnement normal du dépôt où
`langgraph` est déjà installé.

## 3. Produire le template des facteurs

Le nom du dossier V8.3 peut être différent : adapte les chemins.

```powershell
python -m scripts.demo_v84 template `
    --design missions/robustness-design-001/experimental_design.json `
    --matrix missions/robustness-design-001/design_matrix.json `
    --out v84_bindings.json
```

Édite `v84_bindings.json`. Exemples de traitements (les NOMS DE FACTEURS
et de NIVEAUX doivent être les tiens, issus de V8.3) :

```json
{
  "schema_version": "8.4",
  "fixed_model": {"provider": "openai", "model": "MODELE_REEL_EXISTANT"},
  "factor_bindings": {
    "strategy": {"no_retry": "one_shot", "retry_once": "retry_once"},
    "noise": {"clean": "clean", "tool_error": "tool_error"}
  },
  "pricing": {"usd_per_million_input": 1.0, "usd_per_million_output": 5.0}
}
```

Les noms de modèles et prix de l'exemple sont **fictifs/illustratifs**.
Renseigne un modèle actuellement accessible et **ses vrais prix vérifiés**
(USD / million de tokens d'entrée et de sortie), puis vérifie que tes clés
`OPENAI_API_KEY` ou `ANTHROPIC_API_KEY` sont présentes dans `.env`.
N'ajoute jamais de clé au JSON ; il sera archivé avec le plan.

Si ton plan contient un facteur `model_scale`, renseigne un modèle distinct
(`{"provider": "openai", "model": "..."}`) pour CHACUN de ses niveaux,
avec `fixed_model: null`. Un facteur `other`, des niveaux non pris en charge,
ou des données `external_unverified` sont **bloqués** : ne force pas leur
interprétation. Il faudra une implémentation expérimentale dédiée.

Les seuls traitements supportés par ce canary sont :

| Type de facteur V8.3 | Traitements exécutables V8.4 |
|---|---|
| `model_scale` | un modèle/provider concret par niveau |
| `architecture` | `single_agent`, `planner_executor` |
| `orchestration` | `one_shot`, `retry_once` |
| `perturbation` | `clean`, `tool_error` |

Si un facteur est absent, le canary utilise respectivement `single_agent`,
`one_shot`, `clean`, et requiert `fixed_model` s'il n'existe pas de
`model_scale`. Il s'agit de paramètres fixes du canary, non d'une inférence
sur la sémantique de la mission.

## 4. Lancer une cellule sans API (préparation + pause HITL)

```powershell
docker compose -f compose.yaml -f compose.v74.yaml up -d db

python -m scripts.demo_v84 start `
    --thread robustness-canary-001 `
    --design missions/robustness-design-001/experimental_design.json `
    --matrix missions/robustness-design-001/design_matrix.json `
    --bindings v84_bindings.json `
    --cell C001 `
    --evaluations 2
```

`start` vérifie les documents, les niveaux et le budget, puis **s'interrompt**.
Aucun appel LLM expérimental n'est effectué à cette étape.

```powershell
python -m scripts.demo_v84 status --thread robustness-canary-001
```

Relis la cellule et le modèle réellement sélectionnés. Si la correspondance
est erronée, **refuse**, corrige le JSON puis utilise un nouveau thread :

```powershell
python -m scripts.demo_v84 resume --thread robustness-canary-001 --decision reject
```

## 5. Exécution autorisée (appels API PAYANTS)

```powershell
python -m scripts.demo_v84 resume `
    --thread robustness-canary-001 `
    --decision approve
```

Saisis `OUI` après relecture. La commande exécute au plus 4 évaluations sur
une cellule ; le nombre réel d'appels dépend de l'architecture, de la
perturbation et de la stratégie de récupération.

Sortie durable :

```
missions/robustness-canary-001/v84_execution.json
```

Le reçu contient : plan, cellule, modèle utilisé, score exact sur tâches
arithmétiques, compteur de tokens fourni par le SDK, latence et **coût estimé**
à partir du prix saisi par l'opérateur. Il ne contient pas de preuve de
facturation officielle ni d'observations scientifiquement généralisables.

## 6. Limites et précautions importantes

1. **C'est un test d'ingénierie**, pas une évaluation de la robustesse des
   systèmes agentiques telle que pré-enregistrée en V8.3. Le protocole réel,
   les données, la gestion des perturbations, les métriques et la validité
   statistique restent à implémenter/valider avant une publication ACL.
2. Les modèles sont réellement interrogés via les SDK OpenAI/Anthropic, pas
   simulés. Seuls les tests unitaires utilisent un faux provider.
3. Le **budget USD est estimé** d'après les tarifs fournis dans les bindings.
   Les prompts entrants peuvent consommer plus de tokens que la limite
   calculée en préflight ; le programme arrête *après observation* d'un
   dépassement, mais ne garantit pas une limite stricte sur la facture.
4. Le reçu est créé avant les appels. Si le processus s'arrête pendant une
   requête, son état demeure potentiellement inconnu : le système ne réessaie
   pas automatiquement, pour éviter des doubles appels facturables. Une
   intervention humaine et le rapprochement avec les logs du provider sont
   nécessaires. La présence d'un checkpoint ne garantit pas l'**exactly-once**.
5. La perturbation `tool_error` remplace la réponse d'un **calculateur local
   simulant une indisponibilité** ; `retry_once` rejoue ce calcul côté
   orchestrateur. Il ne s'agit pas d'une implémentation d'appel d'outils
   autonome ou d'une récupération de panne réseau générique.
6. L'ancien `experiments/sandbox.py` V7.4 n'est pas modifié. Il sera utile
   dans la prochaine tranche de V8.4 si du code généré doit réellement être
   exécuté ; ce canary n'exécute aucun code arbitraire.
7. Conserve `v84_bindings.json` et `missions/` hors Git si tu ne veux pas y
   publier tes configurations et résultats. Les secrets doivent rester dans
   `.env`, déjà ignoré par Git.

## Critères de fin de cette tranche de V8.4

- [ ] Tests V8.4 + régressions V8.1–V8.3 verts.
- [ ] Mapping explicite V8.3 -> traitement exécutable validé.
- [ ] Approbation/rejet et reprise des checkpoints testés.
- [ ] Une cellule et 1 ou 2 tâches exécutées avec un **vrai** provider.
- [ ] Reçu et métriques observés, sans prétention de contribution scientifique.

**Suite proposée** : brancher de vrais jeux de tâches d'agents, des appels
outils pilotés par le modèle, des contrats d'outils reproductibles, des
comparaisons appariées entre cellules et une politique de coût/erreurs plus
rigoureuse avant de produire des résultats utilisables en V8.5.
