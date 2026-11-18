# Agentic AI Lab — Roadmap et critères de livraison

> **État au 18 novembre 2026.** V8.1 à V8.9 intégrées à `main` via les [PR #3](https://github.com/PaulGru/agentic-ai-lab/pull/3) et [#5](https://github.com/PaulGru/agentic-ai-lab/pull/5). La [CI du commit V8.9](https://github.com/PaulGru/agentic-ai-lab/actions/runs/44274086038) a réussi.

## 1. Jalon atteint — parcours V8.9 intégré

| Bloc | État | Nature de la validation |
|---|---|---|
| V0–V7.4 — fondations agentiques | Fusionné | Tests du dépôt et CI historique |
| V8.1 — Mission & Planning | Fusionné | Contrats, HITL, tests avec provider simulé |
| V8.2 — Literature Review | Fusionné | OpenAlex avec transport simulé en tests ; appels réels effectués localement lors du développement |
| V8.3 — Experimental Design | Fusionné | Contrats et approbations testés |
| V8.4.1–V8.4.6 — moteur expérimental | Fusionné | Backends simulés/local, registre durable, HITL, pilote Docker synthétique testé localement |
| V8.5.1–V8.5.5 — moteur d'analyse | Fusionné | Ingestion vérifiée, descriptif, inférence pédagogique, validité, LangGraph |
| **Handover Codex** | **Fusionné (PR #4)** | Documents de transmission intégrés ; audit en lecture seule effectué avant V8.6 |
| **V8.6 — Scientific Critic & Revision** | Inclus dans la livraison V8.9 | Contrats, preuves vérifiées, journal durable, assistance multi-provider et révisions approuvées |
| **V8.7 — Research Memory** | Incluse dans la livraison V8.9 | Tests de mémoire locale, contrôle PostgreSQL et rappel des décisions lors du pilote |
| **V8.8 — Scientific Paper Writer** | Livré avec le parcours V8.9 | Manuscrit et figure vérifiables depuis les données et décisions |
| **V8.9 — End-to-End Integration & Evaluation** | Livré en mode pilote synthétique | Liaison cellule → job, reprise PostgreSQL, validations humaines et bilan d'exécution |

V8.9 valide le parcours sur des tâches synthétiques à partir d'un protocole
approuvé. La prochaine validation porte sur des observations expérimentales réelles.

## 2. Préparer le pilote V9

Le [protocole du premier pilote](V9_PILOTE_RELANCE_OUTIL.md) fixe le corpus,
le modèle, le budget et les critères de vérification. La calibration attend
sa validation humaine.

- [ ] Définir le protocole pilote et ses critères d'acceptation avant de consulter les résultats.
- [ ] Préparer un workload autorisé et conserver les observations par tâche.
- [ ] Reconstruire l'analyse à partir des traces et évaluer la critique, la mémoire et les validations humaines.
- [ ] Vérifier la provenance des affirmations dans le manuscrit du pilote.

La mémoire conversationnelle de ChatGPT et celle de Codex ne sont pas automatiquement transférées. **Ces fichiers versionnés + les contrats et tests sont la référence durable du projet.**

## 3. V8.6 — Scientific Critic & Revision

**Pourquoi ?** Un système qui sait rédiger un rapport d'analyse sans remettre en question sa méthode peut produire des conclusions trompeuses. Le critique doit détecter les insuffisances, expliquer leur gravité et proposer des révisions vérifiables, tout en préservant l'historique.

### Périmètre implémenté sur la branche V8.6

1. **Contrat de critique structuré** : `CritiqueIssue` (cible, preuve, gravité, type, recommandation) et `ReviewReport` avec provenance / empreintes.
2. **Règles déterministes préalables** : absence d'artefact, incohérence de checksum, réplication insuffisante, métrique non définie, assertions non étayées, confusion simulation/données réelles.
3. **Révision assistée par LLM multi-provider** : propositions explicites et justifiées, sans édition silencieuse des résultats, des mesures ou des citations.
4. **Boucle LangGraph + HITL** : évaluation → proposition → approbation/rejet → nouveau plan/version, avec limite de cycles et traçabilité.
5. **Tests pédagogiques** : rapport synthétique valide, critique mal fondée, référence introuvable, intégrité altérée, reprise après interruption, refus humain et absence d'effets externes.

**Définition de terminé V8.6 :** aucune critique ne transforme des données fictives en faits, tout verdict est rattaché à des preuves, toute révision préserve le document précédent, les tests passent et l'approbation humaine précède les opérations sensibles.

Le détail des sept sous-étapes, des commandes, des tests et des limites figure
dans [V8_6_SCIENTIFIC_CRITIC_REVISION.md](V8_6_SCIENTIFIC_CRITIC_REVISION.md).
La révision approuvée reste prospective. La V8.9 demande une décision
d'exécution séparée et lie explicitement les cellules V8.3 aux seules tâches
arithmétiques du pilote. Le catalogue scientifique reste une étape de V9.

## 4. Après V8.6

| Phase | Objectif | Condition de succès |
|---|---|---|
| **V8.7 — Research Memory & Knowledge Management** | Retrouver les connaissances, épisodes et décisions entre sessions et providers | Sources et projets préservés, souvenirs obsolètes exclus, récupération traçable et bornée ; voir [le guide](V8_7_RESEARCH_MEMORY.md) |
| **V8.8 — Scientific Paper Writer** | Générer un manuscrit structuré et révisable | [Guide du rédacteur](V8_8_SCIENTIFIC_PAPER_WRITER.md) |
| **V8.9 — End-to-End Integration & Evaluation** | Relier toutes les briques en mode synthétique et suivre reprises, décisions et appels | [Guide du parcours et scénarios](V8_9_RESEARCH_WORKFLOW.md) |
| **Backend HPC réel (à planifier)** | Connecter un adaptateur Slurm/SSH au contrat `ExecutionBackend` | Validation avec les règles et accès réels de Jean Zay, tâches minimales approuvées, coûts/quotas contrôlés |
| **V9 — Research Campaign (proposée)** | Utiliser le système pour la vraie question sur la robustesse agentique | Protocole scientifique pré-enregistré, modèles d'étude distincts des agents, réplications contrôlées, statistiques justifiées et revue humaine |

Après V8.9, le catalogue de workloads scientifiques, les modèles de données
pour des expériences réelles, l'évaluation avec des modèles réels et le backend
HPC restent à construire. L'accès Jean Zay et la sélection des modèles étudiés
restent à confirmer pour V9.
