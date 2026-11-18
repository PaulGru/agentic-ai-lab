# Agentic AI Lab — Contexte et décisions architecturales

> **Jalon vérifié le 17 novembre 2026.** Dépôt : `PaulGru/agentic-ai-lab`. Les versions V8.1 à V8.5 ont été intégrées à `main` via la **PR #3**, fusion squash `1283b2c420b2942c950164feff8872a1ea9d3d18`. La CI de la PR sur son dernier commit `e93a846` a réussi ; l'état des futures exécutions CI doit toujours être revérifié. Le dépôt est **public**.
>
> Ce document transmet le contexte métier et les décisions : **il ne constitue pas la transcription de tous les échanges ChatGPT et ne crée pas de mémoire automatique dans Codex**. Le code et les tests de la branche courante restent l'autorité pour le comportement réel.

## 1. Vision du projet

Projet pédagogique Python visant à comprendre les systèmes agentiques modernes par la construction progressive **V0 → V8**. Ambition à terme : un **Autonomous AI Research Scientist** capable de recevoir un brief, proposer une mission, consulter des publications, concevoir des expériences, les exécuter, analyser des observations, critiquer ses conclusions et rédiger un manuscrit scientifique vérifiable (cible de forme ACL, **pas une garantie de publication**).

**Priorité présente : concevoir et valider les briques**, pas répondre à notre future question de recherche. La question de robustesse des systèmes agentiques (influence de la taille, de l'architecture et de l'orchestration) servira de **cas d'étude futur**, éventuellement exécuté sur le supercalculateur Jean Zay après obtention des accès et autorisations nécessaires.

## 2. Principes retenus

- **Pédagogie avant frameworks** : expliquer l'utilité d'une technologie, puis montrer une implémentation minimale et les limites.
- **Multi-provider** : garder les adaptateurs OpenAI et Anthropic interchangeables (`providers/base.py`, `research/factory.py`) ; pas de dépendance scientifique à un fournisseur commercial précis.
- **Contrôle humain** : HITL pour valider plans et effets externes ; budgets et sécurité privilégiés au degré d'autonomie apparent.
- **Local d'abord** : Docker et PostgreSQL pour développer ; cloud externe non indispensable ; accès HPC différé.
- **Traçabilité** : contrats Pydantic, empreintes SHA-256, statuts explicites, idempotence prudente et tests.
- **Preuves vérifiables** : ni métriques ni citations inventées. La provenance doit survivre aux étapes de traitement.

## 3. Trois plans distincts

```text
Control plane (LLM chercheurs)
  providers/ + research/ + LangGraph
      | Mission, littérature, design approuvés
      v
Execution plane (indépendant du fournisseur de l'agent)
  execution/ : ExperimentJob -> HITL -> DurableJobManager -> ExecutionBackend
                                          |                 |           |
                                          |              Docker local  Slurm simulé
                                          v
Data plane
  artefacts / reçus vérifiés -> analysis/ -> descriptif -> validité -> inférence gardée
      |                                                         |
      +--------------------> future scientific critic / paper writer
```

**Distinction centrale** : les LLM OpenAI/Anthropic qui pilotent le scientifique autonome ne sont **pas** les modèles étudiés. Dans l'étude future, les modèles Dense/MoE et différents systèmes agentiques pourront tourner sur Jean Zay. L'architecture interne Dense/MoE d'un modèle n'est pas synonyme d'architecture `single_agent`/`planner_executor` d'un système agentique.

## 4. Historique pédagogique et modules

| Étape | Principales capacités / chemins |
|---|---|
| **V0–V1** | Python, appels LLM, contrats providers, outils, premier agent avec tool calls |
| **V2** | Routage, handoffs, composition de capacités multi-agents |
| **V3** | LangGraph, `StateGraph` et états explicites |
| **V4** | Human-in-the-Loop, interruptions, approbations et reprise |
| **V5** | FastAPI, Docker Compose, PostgreSQL, sécurité de l'API et CI ; déploiement cloud différé |
| **V6** | Recherche web native, audit de métadonnées, pipeline de recherche et stabilisation |
| **V7.1–V7.4** | Propositions d'expérience, sandbox Docker, révision, workflow persistant ; `experiments/` |
| **V8.1** | Brief → mission structurée, empreinte et validation HITL ; `research/mission*` |
| **V8.2** | Catalogue de publications via OpenAlex, recherche bornée, gestion HTTP 429 et revue de littérature ; `research/literature*` |
| **V8.3** | Protocole / matrice de facteurs et validation humaine ; `research/design*`, `research/experimental_design.py` |
| **V8.4.1** | Contrat générique `ExperimentJob` et interface `ExecutionBackend` ; `execution/contracts.py`, `execution/backends/base.py` |
| **V8.4.2** | `LocalDockerBackend`, workload autorisé `synthetic.mean`, exécution isolée et reçus ; `execution/backends/local.py` |
| **V8.4.3** | `SlurmSimBackend` persistant, aperçu `sbatch`, **simulation offline sans HPC** |
| **V8.4.4** | `DurableJobManager` SQLite : registre et soumissions prudentes ; `needs_reconciliation` |
| **V8.4.5** | Orchestration LangGraph et HITL autour des soumissions ; `execution/workflow.py` |
| **V8.4.6** | Pilote E2E : deux jobs Docker calculent deux moyennes synthétiques, collecte vérifiée et rapport ; `execution/campaign*` |
| **V8.5.1–V8.5.5** | Ingestion vérifiée, descriptif, test exact apparié pédagogique, bootstrap, contrôle de validité et graphe LangGraph ; `analysis/` |

Le vieux **canary V8.4** sous `research/execution_contract.py`, `research/execution_graph.py`, `research/canary_runtime.py` et `scripts/demo_v84.py` a été conservé pour l'historique. Il associait des niveaux de recherche à des modèles API du chercheur : **ce n'est plus l'architecture cible**. Le développement doit privilégier `execution/` ; ne pas réintroduire les placeholders `__SET__` du fichier local `v84_bindings.json`.

## 5. Où se trouvent les responsabilités

- `providers/` : adaptateurs LLM ; `agent/` et `orchestration/` : protocoles outils / graphes historiques.
- `api/` : FastAPI et protection `X-API-Key` pour les routes applicatives ; `/health` est public.
- `research/` : missions, littérature, conception expérimentale, pipeline de recherche.
- `experiments/` : V7, sandbox et boucle de révision ; **ne pas autoriser l'exécution arbitraire**.
- `execution/` : contrats, simulation, exécution locale isolée, gestionnaire durable, approbations et campagne pilote.
- `analysis/` : vérification des artefacts, calculs statistiques et validation de leurs préconditions.
- `tests/`, `examples/`, `scripts/`, `docs/` : tests et fixtures, exemples fictifs, démonstrations et explications.
- `runs_control/`, `runs_local/`, `missions/` : sorties locales exclues de Git ; préserver ces données lorsque l'on débogue.

Le document plus ancien `docs/ARCHITECTURE.md` décrit surtout le socle **V6.7** ; le présent document complète l'architecture jusqu'à V8.5. Ne pas supposer que `README.md` décrit exhaustivement V8 (il présente encore principalement V7).

## 6. Persistance, sécurité et provenance

- **LangGraph / PostgreSQL** : checkpoints des workflows persistants, avec `thread_id` ; ne pas modifier les identifiants/nœuds sans plan de migration et tests de reprise.
- **SQLite `DurableJobManager`** : journal de soumission, handle de backend et reçus ; les effets externes ne sont pas exactement-once. En cas de soumission ambiguë, **réconciliation manuelle** avant nouvelle action.
- **Slurm simulé** : gère des états persistants et des artefacts vides marqués `simulated:true`. **Aucun `sbatch`, SSH ou GPU distant** actuellement.
- **Docker local** : seuls les workloads explicitement autorisés sont exécutables sous limites de ressources et en sandbox ; la protection doit être réévaluée avant de traiter du code non fiable.
- **V8.4.6** : deux calculs arithmétiques, moyennes attendues **3.0 et 6.0** ; aucune réplication scientifique.
- **V8.5** : une fixture fictive appariée de dix paires teste les statistiques, mais **ne constitue aucune preuve scientifique**. L'ingestion de métriques réelles du pilote doit relire les jobs/artefacts vérifiés depuis SQLite, et non faire confiance à un JSON isolé.
- **Git** : `.env`, secrets, ZIP, fichiers SQLite, `missions/`, `runs_*` et **`/research_brief.md` personnel** restent exclus. La fixture de test `tests/fixtures/research_brief.md` est committée et non sensible.

## 7. Ce qui manque encore (ne pas surévaluer le projet)

1. V8.9 relie un design approuvé au pilote arithmétique à partir de liaisons explicites. Un catalogue de workloads scientifiques et leurs contrats de traduction restent à construire.
2. Pas de **backend Slurm réel** ni de validation des environnements, quotas, modules, transferts et politiques de Jean Zay. Le simulateur ne vaut pas preuve de compatibilité HPC.
3. Le backend local conserve certaines limites de reprise de workers après interruption ; la gestion des soumissions incertaines est prudente, pas une garantie exactly-once.
4. L'analyse actuelle est pédagogique : agrégats et paires synthétiques ; pas encore de modèles de données pour vraies répétitions multi-factorielles ni de protocole statistique ACL.
5. Le critique V8.6, la mémoire V8.7 et le rédacteur V8.8 sont réunis dans le pilote V8.9. Leur évaluation scientifique avec des modèles et des sources réels relève des travaux suivants.
6. La CI de la PR #3 a été verte pour les tests présents dans le workflow ; cela **ne constitue pas** une validation d'accès Jean Zay, ni de campagne scientifique réelle.

## 8. Commandes et sources de vérité

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s tests -p "test_v85*.py" -v
git status -sb
git log --oneline -5
```

Avec l'environnement Compose configuré :

```powershell
docker compose config --quiet
docker compose build lab
docker compose run --rm --no-deps --entrypoint python lab -m unittest discover -s /app/tests -v
```

La CI actuelle est `.github/workflows/v54-ci.yml` (tests sous Docker + smoke HTTP). Lire aussi `docs/V8_4_6_END_TO_END_EXPERIMENTAL_PIPELINE.md` et `docs/V8_5_COMPLETE.md` avant de toucher aux contrats V8.

## 9. Suite

Le [parcours V8.9](V8_9_RESEARCH_WORKFLOW.md) relie le critique V8.6, la mémoire
V8.7 et le [rédacteur V8.8](V8_8_SCIENTIFIC_PAPER_WRITER.md). Le pilote démarre
depuis les artefacts de conception approuvés. Après validation du manuscrit,
il exporte les sources du pilote, les résultats et les décisions pour examen.
La préparation de la campagne expérimentale V9 figure dans la [feuille de route](ROADMAP.md).

## 10. Branche V8.6 — critique et révision

`research/critique/` ajoute un dossier de preuves reconstruit par V8.5, des
critiques déterministes, des remarques LLM facultatives et non vérifiées
sémantiquement, un journal SQLite distinct et un graphe `v86-critic:`.
Les données et les versions précédentes sont conservées. Une ouverture
explicitement en lecture seule du ledger V8.4 évite toute modification des jobs.

Chaque appel LLM exige une autorisation exacte, avec limites de sortie, délai
et retries désactivés. Une issue incertaine n'est jamais relancée automatiquement.
Une révision du protocole V8.3 est diffée, justifiée, revalidée et soumise à un
autre HITL. Sa version V8.6 est prospective, établie après consultation des
résultats et toujours non exécutée. Aucun lien fictif entre design et résultats
n'est créé. Aucun garde-fou scientifique V8.5 n'est activé ou contourné.

Le [guide V8.6](V8_6_SCIENTIFIC_CRITIC_REVISION.md) explique les sept sous-étapes,
les exemples fictifs, le mode offline, la reprise PostgreSQL, la CI Windows et
les limites. La V8.9 réutilise ce graphe dans son parcours synthétique.

## 11. Branche V8.7 — mémoire de recherche

La mémoire technique des checkpoints reste attachée à un workflow ; les ledgers
conservent les effets et décisions. `research/memory/` ajoute une mémoire de
recherche consultable entre sessions et providers, avec PostgreSQL plein texte
et SQLite/FTS5 offline, sans dépendance supplémentaire.

Des projections déterministes de missions, littérature, designs, observations,
critiques et décisions conservent les sources, leurs empreintes et leurs limites.
Les notes/opinions sont tentatives jusqu'à relecture ; invalidation et remplacement
sont explicites, versionnés et liés au projet. Aucun souvenir n'est une permission
de dépenser ou d'exécuter, ni une preuve scientifique au jalon actuel.

Les planificateurs et l'analyste bibliographique peuvent consulter cette mémoire.
Le critique fige le contexte avant le HITL et indexe ses revues/décisions dans des
nœuds séparés. Une source ou un souvenir sélectionné devenu invalide bloque la
reprise avant tout nouvel appel. Les fonctions historiques restent disponibles
sans mémoire, avec leurs contrats et namespaces existants.

Voir [V8_7_RESEARCH_MEMORY.md](V8_7_RESEARCH_MEMORY.md) pour les commandes, les
tests et les limites. La mémoire précède désormais le Paper Writer V8.8 ; sa
qualité d'aide à la décision scientifique reste à évaluer sur de vrais cas.
