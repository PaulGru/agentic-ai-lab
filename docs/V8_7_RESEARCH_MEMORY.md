# V8.7 — Research Memory & Knowledge Management

La mémoire de recherche conserve et retrouve des connaissances entre missions,
sessions, threads et providers. Elle complète les checkpoints et les journaux
existants. Elle ne modifie aucune mesure et n'accorde aucune permission d'exécuter.

## Rôle des composants

| Composant | Utilité |
|---|---|
| `contracts.py` | Décrit sources, souvenirs, événements de cycle de vie et contexte récupéré. Les identités et empreintes protègent la cohérence du contenu. |
| `sources.py` | Valide les sources V8.1–V8.6 et produit des projections déterministes. Aucun LLM ne décide seul de transformer une sortie en connaissance vérifiée. |
| `stores.py` | Offre PostgreSQL avec recherche plein texte native et SQLite/FTS5 pour le mode offline. Les documents sont immuables ; le cycle de vie est un journal d'événements. |
| `service.py` | Impose le projet, revalide les sources, filtre les souvenirs actifs et borne le contexte. Enregistre ce qui a été réellement récupéré. |
| `integration.py` | Fournit le même contexte aux agents OpenAI ou Anthropic. Les souvenirs sont présentés comme données historiques non fiables, avec leurs références. |
| `demo_v87_memory.py` | Permet l'import, la recherche, l'inspection historique et les décisions humaines sur les notes et versions. |

Aucune dépendance n'est ajoutée. PostgreSQL et psycopg étaient déjà présents ;
SQLite et FTS5 permettent les tests sans serveur ni clé API. Les backends ont le
même contrat de service. Leur présélection plein texte peut différer ; le score
affiché repose sur le recouvrement des termes normalisés et les identifiants
départagent les égalités. Ce score mesure une correspondance textuelle, pas la
vérité scientifique ou la confiance dans une affirmation.

## Ce que la mémoire sait conserver

- **Hypothèses** et question de recherche, avec leur mission d'origine.
- **Plans** et méthodes d'analyse de protocoles approuvés pour la conception.
- **Littérature** : interprétations et extraits contrôlés, toujours limités au
  niveau de vérification des résumés. La présence de l'extrait ne valide pas
  sémantiquement l'interprétation du modèle.
- **Observations** V8.5 : agrégats d'ingénierie ou fixture pédagogique, avec
  conservation du statut synthétique.
- **Décisions** : approbation, rejet ou demande de modification d'un protocole,
  leur commentaire humain et la proposition exacte visée.
- **Enseignements** : limites déterministes du critique et notes proposées.
  Les remarques LLM et notes humaines commencent au statut `tentative`.

Chaque souvenir possède un projet, une catégorie, une clé logique, une source,
un pointeur JSON précis, une empreinte de la valeur référencée, un périmètre et
un niveau de vérification. Les sources sont conservées comme instantanés
immuables, tout en gardant leur origine locale pour une revalidation ultérieure.

Les sources V8.6 sont reconstruites depuis leurs entrées et le ledger original
avant indexation. Un JSON de rapport isolé ne suffit pas. Les autorisations
d'appels API ne sont pas importées comme permissions réutilisables.

## Statuts et versions

Les éléments issus de contrôles de source commencent comme `recorded`. Une
note ou opinion non vérifiée commence comme `tentative` et est exclue du contexte
courant. Une relecture explicite produit `human_reviewed`, sans modifier le
niveau de preuve original. `invalidate` et `supersede` excluent un souvenir des
récupérations futures. Ces événements préservent le texte et les versions.

Le remplacement exige une version active de la même catégorie et de la même
clé logique, dans le même projet. Il n'est pas déduit d'un nom de fichier ou
d'une date. Une version invalidée/remplacée ne peut pas être réactivée. Les
décisions sont transactionnelles ; PostgreSQL les sérialise par projet pour
éviter les remplacements contradictoires concurrents.

Dans ce jalon, tous les souvenirs gardent `scientific_evidence=false` et
`action_authorization=false`. Même une note relue ou une ancienne approbation
ne devient pas une preuve scientifique ou une nouvelle autorisation. Des
contrats pour de vraies preuves expérimentales seront nécessaires avant V9.

## Récupération et traçabilité

La recherche exige un `project_id`. Elle utilise les termes normalisés
(casse/accents, mots fréquents retirés), les catégories et les périmètres.
Les identifiants d'hypothèses et noms de métriques sont aussi indexés comme tags.
Les opérateurs SQL/FTS entrés par l'utilisateur ne sont pas exécutés : la requête
est reconstruite à partir de termes et de paramètres contrôlés.

La présélection est bornée à 50 candidats. Le contexte contient au plus dix
souvenirs, cinq par défaut, et 12 000 octets par défaut (maximum 24 000). Les
extraits sont limités à 700 caractères. Les doublons d'une même connaissance et
version sont écartés du contexte. La recherche reste lexicale : elle ne garantit
pas la découverte de synonymes ou la complétude d'une revue bibliographique.

Avant utilisation, le service vérifie les empreintes, les projections et les
sources originales. Une source absente ou remplacée est exclue avec un motif.
Une altération incohérente échoue explicitement. Le contexte récupéré est
journalisé avec sa requête, ses filtres, ses identifiants et son SHA-256.

`history` peut lire les instantanés historiques sans affirmer que leurs sources
sont encore accessibles. `--verify-source` demande la revalidation actuelle.
Les sources originales et journaux existants restent conservés ; aucune
suppression, aspiration de dossiers privés ou recherche réseau n'est automatisée.

Les frontières de projet sont des règles applicatives, pas une authentification
multi-utilisateur. L'administrateur d'une base locale peut en modifier le
contenu ; un hash n'est pas une signature d'auteur ou une preuve d'identité.

## Intégration aux agents

MissionPlanner, LiteratureAnalyst et ExperimentalDesignPlanner acceptent une
`MemorySession` facultative. Leurs graphes conservent le contexte sélectionné
dans le checkpoint. L'approbation est liée à son empreinte et le service le
revérifie à la reprise. Les namespaces et contrats d'artefacts V8.1–V8.3 restent
inchangés. Les exports déjà produits peuvent être importés explicitement.

Le critique V8.6 ajoute des nœuds de mémoire **uniquement si l'option est activée**.
Il consulte la mémoire avant la revue, fige cette sélection et la joint aux
éventuelles requêtes LLM. L'empreinte de la requête inclut donc les souvenirs
présentés à l'humain. Si un souvenir sélectionné est invalidé ou si sa source
change pendant la pause, l'action est bloquée ; une nouvelle revue est nécessaire.
Les nouvelles connaissances apparues entre-temps n'altèrent pas le contexte
déjà approuvé.

Après enregistrement de la revue et des décisions, un nœud distinct les indexe.
Si l'indexation échoue, les décisions restent conservées et la reprise réessaie
l'indexation idempotente, sans répéter le LLM ni une expérience. Les exports
V8.6 montrent les souvenirs consultés. L'inspection historique du rapport peut
reconstruire sa requête originale depuis cet instantané ; elle ne réactive pas
les souvenirs devenus obsolètes.

La mémoire reste indépendante du provider. Elle ne rajoute ni outil d'exécution,
ni recherche web, ni réparation LLM automatique. Les garanties scientifiques
et HITL de V8.6 demeurent applicables. Le [rédacteur V8.8](V8_8_SCIENTIFIC_PAPER_WRITER.md)
conserve l'empreinte du contexte mémoire consulté lors de la revue.

## Démonstration locale SQLite

Depuis la racine du dépôt, dans l'environnement Python existant :

```powershell
python -m scripts.demo_v87_memory --project demo init
python -m scripts.demo_v87_memory --project demo import-context --path examples/v86_design_context.json
python -m scripts.demo_v87_memory --project demo search --query "robustesse agents" --kind hypothesis
python -m scripts.demo_v86 --memory-db runs_control/v87_memory.sqlite3 --project demo --store runs_control/v87_demo_reviews.sqlite3 offline --input examples/v86_review_input.json --out-dir runs_control/v87_demo_report
python -m scripts.demo_v87_memory --project demo search --query "indépendance réplication" --kind lesson
```

Ces exemples sont entièrement fictifs. Pour une nouvelle revue, utiliser un
nouveau `review_id` dans une copie du fichier d'entrée. Les bases et rapports
restent sous `runs_control/`, exclu de Git. L'option `--db` permet de choisir un
autre fichier mémoire. Les options globales précèdent la sous-commande.

Les imports disponibles sont `import-mission --path`, `import-literature
--mission --literature`, `import-design --mission --literature --design --matrix`,
`import-context --path` et `import-review --store --review`. Tous les fichiers
doivent être fournis explicitement ; ils sont lus en local et validés. UTF-8
avec BOM PowerShell est accepté. Les chemins UNC et URL sont refusés.

Une note peut être proposée avec `note --title ... --text ... --key ... --actor ...`.
Elle reste exclue du contexte jusqu'à `confirm`. Pour confirmer, invalider ou
remplacer, utiliser `--memory-id`, `--target-sha256`, `--actor`, `--reason` et,
pour un remplacement, `--replacement-id`. La CLI affiche le contenu exact et
demande `APPROVE`. `history --memory-id ...` permet de consulter l'état, les
événements et les références. Le SHA-256 de la version visée figure dans
l'objet retourné par `note` ou peut être calculé depuis le contrat `MemoryItem`.

Pour les anciens graphes :

```powershell
python -m scripts.demo_v81 --memory-db runs_control/v87_memory.sqlite3 --project demo start --thread mission-memory-001 --brief tests/fixtures/research_brief.md
```

Les mêmes options existent dans `demo_v82` et `demo_v83`. Les utiliser aussi lors
de la reprise pour vérifier le même projet et la même base. Ces commandes
conservent les coûts/API et validations humaines de leur workflow d'origine ;
elles ne doivent pas être lancées comme des tests offline. Sans ces options,
leur comportement antérieur est conservé.

## PostgreSQL partagé

Configurer `RESEARCH_MEMORY_DATABASE_URL` dans l'environnement ou le `.env`
local ignoré par Git. Choisir explicitement un service approuvé avant la connexion.
Le schéma dédié par défaut est `research_memory_v87`, séparé des checkpoints.
Le DSN n'est jamais sérialisé dans un souvenir, une requête LLM ou un binding.

```powershell
python -m scripts.demo_v87_memory --postgres --project demo init
python -m scripts.demo_v87_memory --postgres --project demo import-context --path examples/v86_design_context.json
python -m scripts.demo_v87_memory --postgres --project demo search --query "réplication"
```

Les graphes acceptent `--memory-postgres --project demo`. Le critique conserve
le binding sans credentials pour les reprises. Les autres processus doivent
configurer le même service et le même projet. Le contrôle d'accès réseau,
l'authentification des utilisateurs et la sauvegarde du service restent à
organiser pour un déploiement partagé en production.

## Tests et limites restantes

```powershell
python -m unittest discover -s tests -p "test_v87*.py" -v
python -m unittest discover -s tests -v
git diff --check
```

Les tests vérifient les redémarrages, la séparation des projets, les notes
tentatives, les remplacements explicites, les artefacts altérés, les budgets,
les filtres et les requêtes hostiles. Ils contrôlent les payloads envoyés à
OpenAI et Anthropic avec des SDK simulés. Un nouvel agent doit retrouver le
motif d'un ancien rejet ; une invalidation pendant une pause doit empêcher
l'appel LLM avant construction du provider.

Contrôle PostgreSQL séparé et explicite :

```powershell
# RESEARCH_MEMORY_TEST_DATABASE_URL ou DATABASE_URL configurée sans l'imprimer
python -m scripts.check_v87_memory
```

Il crée un schéma `v87_test_<uuid>` isolé et préservé, puis vérifie recherche
plein texte, projets, intégration au critique et remplacement dans quatre
processus distincts. L'absence de configuration est un échec, pas un test ignoré.
La CI exécute ce contrôle sous Linux/Docker, ainsi que les tests Python sous
Linux et Windows et les régressions PostgreSQL V8.6.

Validation locale du 18 novembre 2026 : **273 tests réussis sous Windows**, dont
19 tests V8.7. `git diff --check` et `docker compose config --quiet` passent.
Le test PostgreSQL local n'a pas pu aboutir : le moteur Docker Desktop était
arrêté et la connexion au service a expiré. Il faut consulter le résultat du
contrôle PostgreSQL dans la CI du commit publié ; les tests SQLite ne prouvent
pas à eux seuls le fonctionnement PostgreSQL.

L'effet sur la qualité réelle des décisions LLM reste à évaluer sur des cas
annotés par un humain, avec et sans mémoire. Les tests simulés prouvent le
contrat de récupération et les garde-fous, pas un gain scientifique. La recherche
sémantique, la vérification des textes intégraux, les preuves scientifiques
réelles restent des étapes ultérieures.
