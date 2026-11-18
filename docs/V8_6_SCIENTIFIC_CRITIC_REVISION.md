# V8.6 — Scientific Critic & Revision

Nous construisons un scientifique autonome par briques vérifiables. La V8.6
examine des sorties V8.5 et propose des révisions prospectives ; elle n'effectue
aucune campagne scientifique, aucun appel HPC et aucune soumission de job.

## Ce que chaque étape apporte

| Étape | Rôle et raison du changement |
|---|---|
| V8.6.1 — Preuves | `contracts.py` représente le dossier de preuves, les critiques, les propositions et les décisions. `evidence.py` reconstruit les calculs V8.5 depuis leur source. Un `AnalysisReport` isolé ne conserve pas toutes les métadonnées et ne prouve pas sa provenance. |
| V8.6.2 — Règles | `rules.py` relève les limites connues sans LLM : données synthétiques, absence de réplication, hypothèses non démontrées, littérature limitée aux résumés et absence de lien design–exécution. Les portes de validité V8.5 restent l'autorité statistique. |
| V8.6.3 — Mémoire durable | `store.py` conserve les documents et décisions sans écrasement et réserve une intention avant un appel LLM. Un checkpoint LangGraph ne suffit pas à empêcher une double dépense après un incident. |
| V8.6.4 — Assistance LLM | `llm.py` utilise l'interface multi-provider et `AgentRunner`, sans outils ni web natif. Les adaptateurs reçoivent des limites de sortie et de délai, avec retries désactivés. Les valeurs par défaut des usages historiques restent inchangées. |
| V8.6.5 — Révision | `revision.py` calcule un diff, exige une justification par changement et revalide `ExperimentalDesign`. Le planificateur V8.3 seul n'exprime pas la filiation d'une révision motivée par une critique. |
| V8.6.6 — Orchestration | `graph.py` rend explicites les pauses, les décisions et les reprises dans `v86-critic:`. Les sources sont relues après les interruptions. Autoriser un appel LLM et adopter un protocole sont deux décisions distinctes. |
| V8.6.7 — Démonstration | `demo_v86.py`, les exemples fictifs et les tests rendent les comportements observables. `check_v86_postgres.py` vérifie une reprise entre processus avec un vrai checkpointer PostgreSQL. |

Pydantic, LangGraph, SQLite et la bibliothèque standard étaient déjà présents.
Aucune dépendance n'est ajoutée. Pydantic contrôle les structures, SQLite
journalise les décisions et LangGraph orchestre les interruptions ; aucun de
ces outils ne prouve à lui seul la validité scientifique d'un raisonnement.

## Frontières de confiance

Deux sources sont acceptées :

1. `verified_pilot_aggregates` : `PilotProtocol` et ledger V8.4.6 **déjà existant**.
   Le gestionnaire est ouvert avec `read_only=True`, sans backend d'exécution.
   Les métriques, SHA-256, unités et marqueurs de simulation sont revérifiés.
2. `pedagogical_paired_fixture` : fichier fictif validé, puis analyse recalculée
   par V8.5. La valeur-p éventuelle reste strictement pédagogique.

Le mode lecture seule de `DurableJobManager` refuse `submit`, `refresh`,
`cancel`, les mises à jour SQL et la création d'une base manquante. Son mode
historique reste inchangé par défaut.

Le contexte de révision rassemble les quatre documents V8.1–V8.3 dans un objet
`DesignContext` : mission, littérature, design approuvé et matrice. Leur
filiation et leurs empreintes sont contrôlées sans importer le canary V8.4.
Les exports déclarés approuvés restent fondés sur une frontière locale de
confiance : leurs hashes ne sont pas des signatures d'identité.

Le pilote V8.4.6 et un design V8.3 n'ont toujours pas de traduction générique
vérifiée entre eux. Le rapport conserve `design_execution_link=unavailable`
si un contexte est fourni. La révision est donc une revue documentaire du
design, jamais une affirmation que ces résultats en évaluent les traitements.

## Observations, hypothèses et conclusions

- Les observations sont rendues depuis des références internes au dossier ;
  le LLM ne remplace pas leurs valeurs.
- Les hypothèses de mission restent des propositions à tester.
- Les assertions numériques facultatives (`claims`) sont comparées à une
  valeur référencée. Une concordance numérique ne valide pas tout leur texte.
- Les conclusions scientifiques sont refusées sur toutes les sources actuelles.
- Les critiques LLM restent `potential_concern`, avec relecture humaine requise.
  Une référence existante ne démontre pas qu'elle soutient l'interprétation.
  Les commentaires libres peuvent être erronés ; leur vérité sémantique n'est
  pas garantie par le schéma ou les contrôles de références.

Une intégrité invalide arrête le workflow avant interprétation. Une analyse
synthétique correctement calculée peut terminer sa revue tout en conservant
ses blocages scientifiques. Le seuil V8.5 de huit paires n'est pas une étude
de puissance ou une démonstration de l'indépendance des observations.

## Révisions et HITL

Le candidat est un `ExperimentalDesign` V8.3 complet, accompagné d'une
justification pour chaque chemin modifié et d'identifiants de critiques connus.
Le diff est recalculé : dictionnaires de mêmes clés par feuilles, listes en
bloc. Les changements de titre/objectif nécessitent un nouveau processus de
mission/design. Le parent et toutes les mesures restent conservés.

Après approbation, SQLite conserve une enveloppe V8.6 contenant proposition,
décision et matrice régénérée. Elle indique `reviewed_after_results=true`,
`basis=design_review_only_unlinked_results` et `execution_status=not_executed`.
Elle n'est pas un nouvel export V8.3 prétendument pré-enregistré. Aucun adaptateur
ne la transforme automatiquement en jobs. Une exécution future exigera son
propre plan et son propre HITL.

Une passe est autorisée par défaut, deux au maximum. `request_changes` exige
un commentaire humain et conserve le lien vers la proposition précédente.
Au dernier cycle, une nouvelle demande de modification clôture le workflow
avec `cycle_limit` ; elle n'accorde pas une boucle illimitée.

## Commandes offline

Depuis la racine du dépôt, avec l'environnement Python existant :

```powershell
python -m scripts.demo_v86 preview --input examples/v86_review_input.json
python -m scripts.demo_v86 offline --input examples/v86_review_input.json --draft examples/v86_revision_draft.json
python -m scripts.demo_v86 export --review v86-demo-001 --out-dir runs_control/v86_export_001
python -m scripts.demo_v86 inspect --out-dir runs_control/v86_export_001
python -m scripts.demo_v86 verify --review v86-demo-001
```

`preview` n'écrit aucun historique. `offline` interdit les options d'assistance
LLM, utilise un checkpointer en mémoire pendant cette commande et conserve
l'audit SQLite. Avec un draft, saisir `APPROVE` pour adopter la version ; toute
autre réponse la refuse. Sans draft, seule la revue est produite. Pour reprendre
un workflow entre commandes, utiliser le parcours PostgreSQL ci-dessous.

Les chemins du fichier de configuration sont relatifs à la racine du dépôt
puis enregistrés en absolu. Les fichiers UTF-8 avec BOM PowerShell sont acceptés.
Les URL, chemins UNC, JSON surdimensionnés, clés dupliquées et nombres non finis
sont refusés. `--store` précède la sous-commande et permet de choisir un autre
historique. Ne jamais réutiliser un identifiant pour une autre revue.

## Parcours persistant PostgreSQL

Configurer `V86_DATABASE_URL`, ou `POSTGRES_PASSWORD` avec PostgreSQL déjà exposé
sur `127.0.0.1:55432` comme en V8.3. Ces secrets restent dans `.env`. La connexion
et la préparation de cet environnement sont des opérations explicitement
autorisées par l'opérateur ; les commandes ne lancent pas Docker à sa place.

```powershell
python -m scripts.demo_v86 start --input examples/v86_review_input.json
python -m scripts.demo_v86 status --review v86-demo-001
python -m scripts.demo_v86 resume --review v86-demo-001 --decision propose --draft examples/v86_revision_draft.json --actor local-operator
python -m scripts.demo_v86 resume --review v86-demo-001 --decision approve --actor local-operator
```

Le dernier appel affiche la proposition exacte puis demande `APPROVE`.
Utiliser `reject` pour refuser, `skip` à l'entrée de révision pour conserver
seulement le rapport, ou `request_changes --comment "..."` pour demander une
nouvelle proposition dans la limite des cycles.

`status` ne construit aucun provider et ne relit pas les sources ; `verify`
revérifie les fichiers et le ledger d'origine. Le service PostgreSQL doit être
accessible pour lire ses checkpoints. Aucun namespace historique n'est renommé.

## Assistance OpenAI ou Anthropic, facultative et payante

Dans une **copie** du fichier d'entrée, choisir un nouvel identifiant et ajouter :

```json
{
  "llm": {
    "provider": "openai",
    "model": "MODELE_CHOISI_PAR_OPERATEUR",
    "max_output_tokens": 4000,
    "timeout_seconds": 60
  },
  "assist_critique": true,
  "assist_revision": true
}
```

Sélectionner `anthropic` pour l'autre adaptateur et un modèle réellement
accessible. Cet extrait complète la configuration ; il ne constitue pas
un fichier d'entrée complet ni une recommandation de modèle. Les clés sont
chargées depuis l'environnement, jamais depuis le dossier envoyé au modèle.

`start` s'arrête avant l'appel. Chaque `v86_llm_approval` montre la requête exacte,
son provider, son modèle, ses limites et son SHA-256. `resume --decision approve`
nécessite une confirmation `APPROVE`. Après le critique, un autre accord est
requis pour l'éventuelle proposition LLM, puis un troisième pour adopter le
protocole. Il y a au plus un appel critique et deux appels de révision.

Le client applique `max_retries=0`, un délai de 5 à 120 secondes et une sortie
de 256 à 6000 tokens. L'entrée est limitée à 128 KiB, la réponse à 64 KiB.
Ces bornes ne garantissent **pas un plafond USD** ; tokens effectivement
facturés et tarifs ne sont pas centralisés dans cette version (`usage=not_collected`).

Une intention est enregistrée avant l'appel. Un crash ou timeout laisse
`reserved` ou `needs_reconciliation`, sans nouvel appel automatique. Une réponse
invalide laisse `invalid_response`, sans réparation LLM automatique. Examiner
les reçus et les logs du fournisseur avant une décision humaine ; l'application
ne propose pas de remise à zéro automatique. Une réponse déjà conservée comme
`completed` est réutilisée sans provider si le processus s'arrête avant le
checkpoint suivant. `continue --review <id>` tente cette reprise prudente ;
une situation ambiguë reste bloquée. Ne pas modifier les tables pour forcer un retry.

## Exports et intégrité

SQLite contient les versions et décisions originales. L'export contient
`review_package.json`, `review.md` et un `manifest.json` écrit en dernier.
Un fichier divergent ou tronqué n'est pas écrasé : exporter vers un nouveau
dossier depuis SQLite. Un export interrompu avant le manifeste n'est pas complet.

`inspect` vérifie uniquement la cohérence de cet instantané. Il ne prouve pas
l'authenticité des sources ou l'identité de l'opérateur. Les sorties et bases
restent sous `runs_control/`, exclu de Git.

## Validation

```powershell
python -m unittest discover -s tests -p "test_v86*.py" -v
python -m unittest discover -s tests -p "test_v85*.py" -v
python -m unittest discover -s tests -v
git diff --check
```

Les tests utilisent des providers et transports simulés. Les deux adaptateurs
SDK sont vérifiés avec mocks. Aucun test ne coûte de crédit API, n'appelle
OpenAlex ou ne lance de workload Docker. Les nouveaux tests LangGraph ne sont
pas ignorés lorsque la dépendance manque : l'environnement doit être installé.

Validation locale du 17 novembre 2026 : **254 tests réussis sous Windows
Python 3.13**, dont **38 nouveaux tests V8.6**, sans test ignoré. Le contrôle
PostgreSQL ci-dessous a également réussi pour l'approbation et le rejet entre
six processus distincts. Les appels réels OpenAI/Anthropic n'ont pas été lancés.

Contrôle d'intégration **séparé**, avec un PostgreSQL approuvé :

```powershell
# V86_TEST_DATABASE_URL ou DATABASE_URL configurée sans l'imprimer
python -m scripts.check_v86_postgres
```

Il crée un schéma neuf `v86_test_<uuid>` et vérifie approbation et rejet dans
six processus Python distincts. Les schémas existants et volumes sont préservés.
Le schéma de test reste présent ; aucune suppression n'est automatisée. Une
configuration absente provoque un échec explicite, pas un test silencieusement
ignoré. La CI exécute ce contrôle dans son PostgreSQL isolé, les tests Python et
HTTP sous Linux/Docker, et les tests Python sous Windows.

Ce dispositif ne valide toujours ni des appels LLM réels, ni le backend Docker
expérimental réel, ni Slurm/Jean Zay, ni une méthodologie de campagne scientifique.
