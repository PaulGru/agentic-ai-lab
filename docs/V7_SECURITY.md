# Contrat de sécurité de la V7 — Experiment Agent

## V7.1 : générer, ne pas exécuter

Le premier Experiment Agent doit proposer un protocole reproductible,
une hypothèse, du code Python et des dépendances explicites. **Aucun** code
produit par le LLM ne doit être exécuté automatiquement à cette étape.

## V7.2 : exécuter dans un environnement distinct

Le conteneur `api` conserve les clés LLM et l'accès PostgreSQL : il ne doit
jamais servir de sandbox d'exécution du code généré. Créer un environnement
d'exécution dédié avec une frontière de permissions explicite :

- **Aucun secret :** ne pas transmettre `.env`, les clés API, les credentials
  PostgreSQL, les fichiers SSH ou les tokens au processus exécutant.
- **Aucun accès au démon Docker :** ne jamais monter `/var/run/docker.sock`.
- **Pas de réseau par défaut :** les recherches Internet sont faites par
  l'agent de recherche, avant l'exécution. L'accès sortant éventuel doit
  être soumis à une politique explicite.
- **Moindres privilèges :** utilisateur non-root, système de fichiers racine
  en lecture seule si possible, dossier de sortie temporaire isolé,
  capacités Linux retirées, `no-new-privileges`, règles seccomp/AppArmor.
- **Quotas :** durée maximale, CPU, RAM, processus, stockage, volume des logs,
  taille des fichiers de sortie et nombre de tentatives limités.
- **Validation et approbation :** inspecter le plan et les effets proposés ;
  exiger une approbation humaine pour les opérations ayant un effet sensible.
- **Reproductibilité :** versions de dépendances fixées, graine aléatoire
  contrôlable, entrées et artefacts de sortie identifiés.
- **Audit :** consigner le code exécuté, la configuration d'exécution,
  le statut, la durée et les artefacts sans enregistrer de secrets.

Un simple conteneur Docker réduit les risques mais **n'est pas une garantie
absolue d'isolation** face à du code intentionnellement malveillant. La portée
et le modèle de menace doivent être précisés avant d'autoriser du code non fiable.

## V7.3 et suivantes

Définir des tentatives bornées et idempotentes ; ne pas permettre à l'agent
de contourner les règles du sandbox en réécrivant la configuration d'exécution.
Traiter le stdout/stderr et les fichiers produits comme des données non fiables.

## Contrat pédagogique

Le futur Experiment Agent n'aura initialement accès qu'à un outil structuré,
par exemple `prepare_experiment`, puis à un service `run_experiment` à politique
stricte. La séparation génération / validation / exécution / analyse doit
rester visible dans LangGraph.
