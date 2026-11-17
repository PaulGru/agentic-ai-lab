# V8.3 — Experimental Design (Agentic AI Lab)

## Objectif et frontières

La V8.3 transforme les artefacts approuvés de **V8.1** et **V8.2** en un **protocole expérimental pré-spécifié** de type *exploratory pilot*. Aucune expérience ne tourne ; aucun code LLM n'est exécuté ; aucun dataset, licence ou modèle externe n'est déclaré vérifié. Un protocole approuvé n'établit **ni la nouveauté, ni un résultat, ni la puissance statistique** nécessaire à un article ACL.

Le système réutilise `AgentRunner`, `ToolRegistry()` vide, `create_provider(native_web_search=False)` (pour Anthropic, V8.3 augmente localement la limite de sortie à 6 000 tokens pour le JSON), Pydantic, LangGraph, les interruptions HITL et PostgreSQL. Il n'ajoute **aucune nouvelle dépendance**.

## Nouveaux fichiers

- `research/experimental_design.py` : types Pydantic et contrôles déterministes (facteurs, contrastes, baselines, références, estimations et limites, matrice).
- `research/design_planner.py` : planificateur LLM multi-provider, sans appels d'outils ni exécution.
- `research/design_graph.py` : `draft -> approval`, nouveau namespace de checkpoint.
- `scripts/demo_v83.py` : commandes CLI `start`, `status`, `resume`, `export`.
- `tests/test_v83_design.py` : tests sans réseau/LLM réel, plus reprise HITL en mémoire.

## Prérequis

1. V8.1 approuvée : `missions/robustness-001/mission.json`.
2. V8.2 approuvée : `missions/robustness-search-003/literature.json` (le chemin dépend du `--thread` utilisé pour **V8.2**).
3. Environnement virtuel avec les dépendances déjà installées, `.env` avec `POSTGRES_PASSWORD`, le provider et son modèle configurés.
4. PostgreSQL exposé localement : `docker compose -f compose.yaml -f compose.v74.yaml up -d db`.

Le contenu des fichiers d'entrée et leurs SHA-256 doivent correspondre. La V8.2 doit comporter `status=approved` et le niveau de vérification des résumés attendu.

## Exécution PowerShell

```powershell
python -m unittest discover -s tests -p "test_v83_design.py" -v
python -m unittest discover -s tests -v

python -m scripts.demo_v83 start `
  --thread robustness-design-001 `
  --mission missions/robustness-001/mission.json `
  --literature missions/robustness-search-003/literature.json

python -m scripts.demo_v83 status --thread robustness-design-001

# Après lecture de tout le protocole ; confirmation exacte OUI :
python -m scripts.demo_v83 resume `
  --thread robustness-design-001 `
  --decision approve

# Si besoin de réexporter le même contenu déjà approuvé :
python -m scripts.demo_v83 export --thread robustness-design-001
```

La commande `resume --decision reject` clôture le thread sans export. Pour proposer un **autre** protocole, démarrer un **nouveau thread**. Ne pas écraser les artefacts existants.

## Sorties après approbation

```
missions/robustness-design-001/
  experimental_design.json
  experimental_design.md
  design_matrix.json
```

`experimental_design.json` trace les empreintes des entrées, le plan validé et la taille prévue des évaluations. `experimental_design.md` est destiné à la relecture scientifique. `design_matrix.json` énumère **toutes** les cellules factorielles et documente `tasks_per_cell` et `seeds` ; il ne crée pas de résultats.

## Principes expérimentaux

- Chaque hypothèse de V8.1 doit posséder un contraste entre **deux niveaux réels d'un facteur**, un estimand, une hypothèse nulle et une règle de réfutation. Les autres facteurs doivent être marginalisés ou maintenus à des niveaux déclarés.
- Une baseline indique **tous les niveaux** des facteurs (comparaison contrôlable).
- Une étude appariée compare les conditions avec les **mêmes tâches et graines**. Les conditions de perturbation comportent nécessairement un contrôle `clean`.
- Métriques minimales : `task_success`, `total_tokens`, `latency_ms`; il faut définir unités, agrégation et traitement des échecs ou données manquantes.
- L'agent pré-déclare analyse d'incertitude, comparaisons multiples, critères d'arrêt et limites de puissance.
- Les liens bibliographiques autorisés concernent uniquement des `work_id` présents dans les *findings* de V8.2, **non** n'importe quelle notice. Ce contrôle ne prouve pas la validité de l'interprétation scientifique.
- Les limites maximales du pilote sont **24 cellules**, **240 évaluations**, **2 000 appels LLM planifiés**, **12 millions de tokens planifiés** et **50 USD d'enveloppe déclarée**. Ce sont des garde-fous **du contrat**, **pas un mécanisme d'application du plafond financier**. Avant V8.4, il faudra intégrer un *metering* d'appels/tokens/coûts réellement facturés.
- Les modèles, licences, accès à des données, datasets, tarifs et capacité d'exécuter le protocole sont **à confirmer** par un humain avant V8.4.

## Validation humaine (HITL)

Le graphe enregistre l'empreinte SHA-256 canonique de `mission.json`, `literature.json` et du design. La reprise vérifie les empreintes. Un SHA-256 protège contre les modifications accidentelles dans ce flux ; **ce n'est pas une signature d'identité ou une preuve d'approbation indépendante**. Les données externes restent non fiables ; elles ne peuvent pas imposer de nouveaux outils ou des étapes d'exécution.

## Reproductibilité / ACL

L'ARR Responsible NLP checklist demande notamment de décrire les limites, données et licences, configurations expérimentales, budget de calcul et statistiques descriptives (par exemple répétitions ou intervalles). Voir :

- https://aclrollingreview.org/responsibleNLPresearch/
- https://ethics.aclweb.org/resources/ethical-review-recommendations/

V8.3 prépare ces champs. Les vérifier et les remplir sur **des expériences exécutées** sera le travail de V8.4–V8.6.

## Limites connues

- Le plan est une **proposition produite par un LLM** ; Pydantic détecte des incohérences structurelles, **pas** la pertinence d'une expérience ni une revendication de causalité.
- V8.2 n'a vérifié que des extraits de résumés ; lire les articles intégraux et contrôler la nouveauté reste indispensable.
- Le modèle peut produire un JSON invalide ou un plan trop ambitieux. Les erreurs de validation sont visibles et aucun protocole n'est approuvé automatiquement ; une nouvelle génération utilise un nouveau thread.
- Les plafonds de coût, de tokens et les versions de modèle ne sont pas techniquement exécutés en V8.3.
- Le nombre d'évaluations est un **plafond de planification** : produit de toutes les cellules, tâches et graines, non un estimateur de puissance statistique.
