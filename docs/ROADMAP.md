# Agentic AI Lab — Roadmap et critères de livraison

> **État au 17 novembre 2026 :** V8.1–V8.5 fusionnées dans `main` via [PR #3](https://github.com/PaulGru/agentic-ai-lab/pull/3), commit squash `1283b2c`. La CI de cette PR a réussi sur `e93a846`. Vérifier la branche locale et les nouvelles CI avant chaque intervention.

## 1. Jalon atteint : plateforme V8.5 consolidée

| Bloc | État | Nature de la validation |
|---|---|---|
| V0–V7.4 — fondations agentiques | Fusionné | Tests du dépôt et CI historique |
| V8.1 — Mission & Planning | Fusionné | Contrats, HITL, tests avec provider simulé |
| V8.2 — Literature Review | Fusionné | OpenAlex avec transport simulé en tests ; appels réels effectués localement lors du développement |
| V8.3 — Experimental Design | Fusionné | Contrats et approbations testés |
| V8.4.1–V8.4.6 — moteur expérimental | Fusionné | Backends simulés/local, registre durable, HITL, pilote Docker synthétique testé localement |
| V8.5.1–V8.5.5 — moteur d'analyse | Fusionné | Ingestion vérifiée, descriptif, inférence pédagogique, validité, LangGraph |
| **Handover Codex** | **Fusionné (PR #4)** | Documents de transmission intégrés ; audit en lecture seule effectué avant V8.6 |
| **V8.6 — Scientific Critic & Revision** | **Implémenté sur branche dédiée, non fusionné** | Contrats, preuves vérifiées, règles, journal durable, assistance multi-provider, révisions et HITL ; voir le guide V8.6 |
| **V8.7 — Research Memory** | **Implémentée sur branche dédiée, non fusionnée** | 273 tests Windows réussis, dont 19 tests mémoire ; contrôle PostgreSQL séparé dans la CI |

**Attention :** « fusionné » ne veut pas dire « prêt pour une recherche ACL » : la chaîne est encore partiellement composée de démonstrateurs indépendants, et aucun essai HPC réel n'a été exécuté.

## 2. Check-list du relais vers Codex

- [x] Fusionner la V8.1–V8.5 dans `main` après CI réussie (PR #3).
- [x] Corriger la fixture V8.1 indépendante du brief scientifique privé.
- [x] Relire et fusionner la PR contenant les trois documents de transmission (PR #4).
- [ ] Mettre à jour le dépôt local avec `git switch main` puis `git pull --ff-only origin main` **après** vérification de `git status`.
- [ ] Vérifier que le dépôt local n'a ni secrets ajoutés ni modifications non enregistrées.
- [ ] Installer/ouvrir l'extension Codex officielle dans VS Code et ouvrir **la racine de ce dépôt**.
- [x] Confier à Codex un **audit en lecture seule** avant tout développement V8.6.
- [ ] Comparer cet audit aux contrats réels et corriger les incohérences documentaires avant de démarrer les nouvelles fonctionnalités.

La mémoire conversationnelle de ChatGPT et celle de Codex ne sont pas automatiquement transférées. **Ces fichiers versionnés + les contrats et tests sont la référence durable du projet.**

## 3. Prochaine étape fonctionnelle : V8.6 — Scientific Critic & Revision

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
L'approbation d'une révision n'autorise aucun job ; le lien générique entre
design V8.3 et exécution reste à construire. L'inférence scientifique reste
interdite sur les sources synthétiques actuelles. Une fusion dans `main`
nécessitera une décision distincte de la publication de la branche.

## 4. Après V8.6

| Phase | Objectif | Condition de succès |
|---|---|---|
| **V8.7 — Research Memory & Knowledge Management** | Retrouver les connaissances, épisodes et décisions entre sessions et providers | Sources et projets préservés, souvenirs obsolètes exclus, récupération traçable et bornée ; voir [le guide](V8_7_RESEARCH_MEMORY.md) |
| **V8.8 — Scientific Paper Writer (renumérotée)** | Générer un manuscrit scientifique structuré et révisable | Affirmations liées aux sources/artefacts ; limites et absence de données réelles explicitement indiquées |
| **V8.9 — End-to-End Integration & Evaluation (proposée)** | Relier toutes les briques et mesurer autonomie, coûts, fiabilité, reprises et qualité | Tests négatifs et scénarios reproductibles, sans dépendance obligatoire à HPC |
| **Backend HPC réel (à planifier)** | Connecter un adaptateur Slurm/SSH au contrat `ExecutionBackend` | Validation avec les règles et accès réels de Jean Zay, tâches minimales approuvées, coûts/quotas contrôlés |
| **V9 — Research Campaign (proposée)** | Utiliser le système pour la vraie question sur la robustesse agentique | Protocole scientifique pré-enregistré, modèles d'étude distincts des agents, réplications contrôlées, statistiques justifiées et revue humaine |

La mémoire V8.7 est insérée avant la rédaction, à la suite de la revue de la
feuille de route. V8.8, V8.9, le backend HPC et V9 restent à construire.
L'accès Jean Zay et la sélection des modèles évalués sont encore à confirmer.

## 5. Premier message conseillé à Codex

> Lis `AGENTS.md`, `docs/PROJECT_CONTEXT.md` et `docs/ROADMAP.md`, puis inspecte le dépôt et ses tests. Fais un **audit en lecture seule** de l'architecture V8.1–V8.5, de ses limites et de la préparation à V8.6. Signale les écarts entre documentation et code, et propose un plan minimal de V8.6 avec contrats, tests et points d'approbation. **Ne modifie aucun fichier et n'exécute aucune action payante ou destructive avant mon approbation.**
