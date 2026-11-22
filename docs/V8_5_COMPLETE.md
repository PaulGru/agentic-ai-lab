# Agentic AI Lab — V8.5 complète : Results & Statistical Analysis

**Périmètre : construction des briques, pas réalisation de la recherche scientifique.**
La V8.5 ne lance ni LLM, ni Docker, ni job Slurm. Elle exploite une campagne V8.4.6 **déjà terminée** ou un jeu de données **explicitement fictif**.

## Pourquoi ces technologies ?

- **Pydantic** (déjà présent en V8.4) décrit et vérifie les contrats. Il empêche des observations incohérentes, des nombres non finis et l'élévation mensongère de `scientific_evidence`.
- **`statistics`, `random`, `itertools` de Python** réalisent les calculs de manière déterministe, auditables sans ajouter SciPy à ce stade.
- **LangGraph** (déjà utilisé depuis V3) rend explicite l'orchestration *ingestion → qualité → descriptif → inférence gardée → rapport*. Il n'ajoute aucun appel LLM et ne reprend pas le rôle de l'HITL de V8.4 : ici, l'analyse est en lecture seule.
- **SQLite V8.4.4** conserve le journal fiable des jobs. Pour les données réelles du pilote, les métriques doivent être reconstruite via `ingest_verified_pilot` à partir de cette base, jamais simplement depuis un rapport JSON non vérifié.

## Les étapes livrées

| Étape | Module | Fonction |
|---|---|---|
| V8.5.1 | `analysis/contracts.py`, `analysis/ingest.py` | Ingestion vérifiée des agrégats V8.4.6 |
| V8.5.2 | `analysis/descriptive.py` | Moyenne, médiane, minimum, maximum, écart-type si applicable, contraste arithmétique |
| V8.5.3 | `analysis/inference.py` | Test exact bilatéral par inversion de signes (2^n combinaisons), IC bootstrap percentile illustratif, effet standardisé apparié |
| V8.5.4 | `analysis/validity.py`, `analysis/stat_contracts.py`, `analysis/pipeline.py` | Validations, refus de pseudo-réplication, refus des inférences hors périmètre, export avec empreinte |
| V8.5.5 | `analysis/graph.py` | Graphe LangGraph sans effet externe, mêmes résultats que le pipeline Python direct |

## Deux jeux de données volontairement séparés

### A — Agrégats V8.4.6 réellement calculés et vérifiés

- **Source autorisée** : `PilotProtocol` + `DurableJobManager` (SQLite), via `ingest_verified_pilot` qui recalcule et contrôle l'intégrité des métriques et des jobs.
- Un job = **une observation agrégée**. Cinq valeurs dans `synthetic.mean` ne constituent **pas cinq répétitions indépendantes**.
- Résultat : **statistiques descriptives seulement**. `inference = null`, `scientific_evidence = false`.

```
python -m scripts.tutorials.paired_analysis pilot --protocol examples/v846_protocol.json --jobs-db runs_control/v846_jobs.sqlite3
```

Si V8.4.6 n'a pas été exécutée et approuvée, terminer d'abord la campagne avant cette commande.

### B — Fixture synthétique appariée (uniquement pour tester l'algorithme)

- **Source** : `examples/v85_paired_fixture.json` — dix paires de scores entièrement inventés.
- Le test exact calcule une valeur-p bilatérale sous l'hypothèse que l'inversion des signes est permutable sous le nul ; cette hypothèse n'est **pas démontrée** par le seul fichier JSON.
- Le bootstrap utilise 2 000 rééchantillonnages avec `seed=42` pour garantir la reproductibilité. Il s'agit d'une **illustration**, pas d'une méthode de confiance validée sur une étude réelle.
- La porte de validité refuse l'inférence pédagogique si moins de huit paires ou si les écarts sont constants ; les doublons et valeurs non finies sont invalides.
- Les rapports affichent explicitement `scientific_evidence = false` et `ready_for_scientific_inference = false`, même si une valeur-p est disponible.

```
python -m scripts.tutorials.paired_analysis fixture --fixture examples/v85_paired_fixture.json
```

Pour les données fournies, résultat pédagogique attendu : moyenne des différences `1.9`, valeur-p exacte `0.00390625` (1024 inversions), intervalle bootstrap indicatif `[1.2, 2.6]`. **Ces chiffres ne documentent aucun modèle réel.**

## Démonstration LangGraph (optionnelle)

Les deux modes supportent `--graph`. L'option utilise un `MemorySaver` **éphémère**, uniquement pour la démonstration ; on peut injecter un checkpointer durable compatible dans `AnalysisWorkflow.build(saver)` lors d'une intégration ultérieure.

```
python -m scripts.tutorials.paired_analysis fixture --fixture examples/v85_paired_fixture.json --graph
python -m scripts.tutorials.paired_analysis pilot --protocol examples/v846_protocol.json --jobs-db runs_control/v846_jobs.sqlite3 --graph
```

**Aucune approbation HITL supplémentaire** : ce graphe ne soumet pas de job. L'exécution sous V8.4.6 reste soumise à approbation humaine.

## Sorties

Par défaut dans `runs_control/v85_analysis/<dataset-id>/` :

- `analysis_report.json` : description, contrôle de validité, éventuellement inférence, provenance et garde-fous.
- `analysis_report.sha256` : empreinte du modèle canonique de rapport. **Checksum d'intégrité, pas signature cryptographique d'authenticité**.

```
python -m scripts.tutorials.paired_analysis inspect --out-dir runs_control/v85_analysis/v85-synthetic-paired-001
```

`inspect` vérifie la cohérence d'un export existant ; pour revalider la provenance d'une campagne V8.4.6, il faut relancer `pilot` avec le journal SQLite.

Les exports sont idempotents et refusent l'écrasement d'un rapport divergent. Les fichiers n'offrent pas encore une transaction atomique multi-fichiers ; une interruption en cours d'export peut nécessiter une inspection ou une reprise manuelle.

## Tests

Depuis la racine du dépôt (Python 3.12/3.13, Pydantic v2, LangGraph déjà installé côté utilisateur) :

```
python -m unittest discover -s tests -p "test_v85*.py" -v
python -m unittest discover -s tests -v
```

Les tests de l'interface LangGraph sont ignorés avec un message explicite lorsque la bibliothèque n'est pas installée : il faut les voir **réussir**, et non *skipped*, sur ton environnement.

## Limites assumées et prochaine étape

- Pas d'étude ACL, pas de contrôle de facteurs multiples, pas de comparaisons multiples, pas d'analyse de puissance ni de causalité.
- Les scores de la fixture sont totalement synthétiques. Les agrégats V8.4.6 n'ont aucune réplication scientifique.
- Pour utiliser un vrai jeu de données issu de Jean Zay, il faudra étendre les contrats et valider provenance, identités des tâches, randomisation, conditions, appariement, répétitions et hypothèses avant d'activer l'inférence scientifique.
- **Fin de V8.5** : effectuer un passage de stabilisation Git/CI et préparer `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/ROADMAP.md` avant de transmettre le développement à Codex.
