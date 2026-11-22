# Du brief au manuscrit

Le point d'entrée `scripts.research_workflow` prend un brief Markdown une seule
fois. Le workflow reprend la mission, la découverte OpenAlex et la synthèse
V8.1–V8.3. Il prépare ensuite un protocole et vérifie la présence d'un exécuteur
compatible. Un journal SQLite local conserve chaque version, les approbations
et la réservation qui précède chaque effet externe. LangGraph reste utilisé
par les composants existants, dont la critique du pilote.

## Comment le scientifique travaille à partir du brief

Le brief Markdown contient une question de départ, son contexte, un périmètre,
des contraintes et des critères de réussite. La question peut rester large.
L'agent propose une formulation mesurable, des hypothèses réfutables et des
requêtes bibliographiques à partir des pistes fournies. La revue humaine de la
mission est le premier moment pour préciser la question ou demander une révision.

| Étape | Ce que fait le système |
|---|---|
| Mission | Le [planificateur](../research/mission_planner.py) reçoit les sections du brief et des consignes qui lui demandent de préciser les hypothèses, les observations qui les réfuteraient, les variables et les risques. Python valide la réponse avant sa revue. |
| Bibliographie et cadre théorique | Après autorisation, Python rassemble des notices OpenAlex. L'[analyste](../research/literature_analyst.py) reçoit la mission et ces notices. Il relie ses constats à des extraits de résumés, signale les lacunes possibles et propose des questions à approfondir. Le programme vérifie chaque extrait dans la notice. Cette synthèse exploratoire peut servir de point de départ à un cadre théorique. Les arguments qui s'appuient sur les méthodes ou résultats détaillés demandent une lecture des articles complets. |
| Protocole | Le [planificateur expérimental](../research/design_planner.py) reçoit la mission et la synthèse approuvées. Il propose une matrice de facteurs, des contrôles, des critères de mesure, une analyse et un budget. Pour les profils de pilote prédéfinis, le workflow prépare directement le protocole compatible et affiche la question effectivement mesurée. |
| Expériences et résultats | L'exécuteur compatible réalise les essais après autorisation et journalise leurs reçus. Pour `retry_catalog`, l'[analyse](../analysis/retry_pilot.py) vérifie les traces d'origine, recalcule les scores et compare les conditions par tâche. Le fournisseur du scientifique et les modèles évalués sont sélectionnés à deux endroits distincts. |
| Discussion, conclusions et manuscrit | Le critique relève les limites, puis le [rédacteur](../research/manuscript.py) assemble les observations, la discussion, la figure et les références disponibles. Les affirmations renvoient aux preuves d'origine. Le brouillon Markdown reprend la structure d'un article scientifique, avec la mise en forme ACL comme cible éditoriale. L'opérateur approuve la version à exporter. |

### D'où viennent les consignes et le contexte

Ce guide explique le processus à l'équipe. Pendant une exécution, les
consignes destinées au LLM sont versionnées dans le code, par exemple
`MISSION_INSTRUCTIONS`, `LITERATURE_INSTRUCTIONS` et `DESIGN_INSTRUCTIONS`.
Chaque appel assemble la consigne de son rôle, les données utiles à cette
étape et le format de sortie attendu. Une révision peut aussi transmettre
le retour humain à l'agent concerné. Les contrats Pydantic vérifient la
structure et les validations Python contrôlent, selon l'étape, les liens vers
les sources, les hypothèses ou les limites du protocole.
Le brief et les sources sont des données pour ces appels. Quand la méthode
évolue, la consigne et les contrôles de l'étape concernée définissent le
nouveau comportement attendu.

La [mémoire de recherche](V8_7_RESEARCH_MEMORY.md) indexe des éléments traçables
tels que les missions, les synthèses et les constats de revue. Les composants
qui consultent cette mémoire sélectionnent des éléments pertinents à partir de
leurs sources. Le journal SQLite du brief conserve, lui, l'état courant,
les versions et les décisions nécessaires pour reprendre le parcours.

### Ce que fait l'orchestration

Le [workflow du brief](../research/brief_workflow.py) utilise des transitions
Python et le journal SQLite pour décider de l'étape suivante, attendre une
autorisation ou proposer un retour en arrière. LangGraph organise des
sous-workflows comme la [critique](../research/critique/graph.py), avec des
étapes et des transitions définies dans le code. La méthode scientifique vient
des consignes, des contrôles, des sources et des décisions humaines. Le
parcours enchaîne les étapes et sollicite l'opérateur aux points de décision
prévus.

### Périmètre vérifié

- `retry_catalog` va jusqu'au manuscrit descriptif du pilote, avec une
  bibliographie issue de la revue approuvée et des résultats reconstruits à
  partir des essais.
- `synthetic.mean` va jusqu'à un rapport d'intégration Docker sur des listes
  fictives.
- `ood_replay` va jusqu'au manuscrit après une répétition avec des réponses
  préenregistrées. Les scores et les conclusions de ce test sont marqués comme
  simulés.
- `ood_gpu` vérifie le transport SSH et l'inférence GPU sur quatre cas fixes de
  calibration. Pour mesurer l'effet d'une relance après une panne transitoire de
  lecture, le profil `retry_gpu` construit son lot à partir du protocole approuvé
  et du corpus fourni.
- Pour une nouvelle question, le workflow peut préparer une mission, une
  synthèse exploratoire et un protocole. Le sélecteur peut choisir un corpus
  compatible dans le catalogue, puis le workflow confie l'implémentation à
  l'agent de code. L'option `--evaluation` permet d'utiliser un corpus fourni.

## Répéter le parcours OOD hors ligne

Le [brief fictif](../tests/fixtures/ood_replay_brief.md) demande de comparer la
formulation de référence et une reformulation de deux tâches appariées. Le
profil `ood_replay` relie le protocole approuvé à huit jobs simulés, sur deux
modèles fictifs. L'analyse vérifie les réponses dans le journal SQLite, recalcule
les scores exacts et transmet les écarts au critique et au rédacteur. Le modèle
simulé A passe de 2/2 à 1/2, soit 50 points de baisse. Le modèle simulé B obtient
1/2 dans les deux conditions.

```powershell
python -m unittest tests.test_ood_replay_workflow -v
```

Cette commande simule aussi la planification et les notices OpenAlex. Elle
vérifie la revue du protocole et l'autorisation d'exécution séparées, le
rattachement des reçus après interruption et le blocage sur un reçu manquant
ou altéré. Les approbations ciblent une version précise. Le brouillon
Markdown renvoie aux scores, dont les identifiants et empreintes renvoient
aux jobs du journal. Ces résultats servent à contrôler le workflow avant
de tester des modèles locaux.

## Préparer la calibration GPU de Qwen

Le profil `ood_gpu` reprend les deux tâches appariées et exécute quatre
générations dans un seul lot, sur le GPU choisi. [OpenSSH](../execution/ood_gpu.py)
transporte les entrées JSON vers un programme fixe. Sur Linux, le
[programme GPU](../execution/ood_gpu_worker.py) utilise PyTorch pour exécuter
Qwen en BF16 et Transformers pour charger le tokenizer et générer les réponses.
L'approbation affiche la cible, la carte, la révision, l'empreinte du programme
et les entrées exactes. Le lot a dix minutes pour se terminer. Chaque entrée
est plafonnée à 512 tokens et chaque génération à 16 tokens.

Préparer séparément sur la machine autorisée un environnement Python avec
PyTorch compatible CUDA, Transformers (au moins 4.51) et les poids Safetensors
de [Qwen3-4B-Instruct-2507 à la révision retenue](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554).
Copier `execution/ood_gpu_worker.py` dans le dossier du pilote. Le programme
charge la révision depuis sa copie locale, en mode hors ligne et avec le
chargement de code distant désactivé. La cible SSH doit être utilisable en mode
non interactif, avec une clé d'hôte déjà vérifiée. Les chemins du serveur et
l'alias restent dans la configuration locale.

Exemple de configuration à adapter dans `runs_control/gpu-config.json`.

```json
{
  "ssh_host": "gpu-lab",
  "python_path": "/home/operator/lab-gpu/.venv/bin/python",
  "worker_path": "/home/operator/lab-gpu/ood_gpu_worker.py",
  "snapshot_path": "/home/operator/models/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
  "gpu_index": 0
}
```

Pour une nouvelle mission, choisir `--workload ood_gpu --gpu-config
runs_control/gpu-config.json` à `start`. Pour une mission existante, utiliser
ces options avec `revise --stage protocol` et le SHA du checkpoint affiché
par `status`. Le modèle du scientifique se sélectionne toujours avec
`--provider` et `--model`. Après revue du protocole, le mode `guide` demande
l'autorisation du lot SSH avec ses limites. Cet essai utilise un modèle
et deux tâches pour vérifier le raccord. La calibration de référence est
réussie si les deux réponses de référence correspondent aux valeurs attendues.

Le reçu conserve les réponses brutes, la durée, les comptes de tokens, le
pic de mémoire allouée par PyTorch et les versions du runtime. L'analyse
recalcule l'exactitude et alimente la critique et le manuscrit. Si SSH échoue
après réservation, l'état demande une réconciliation. `recover` peut
rattacher un reçu complet déjà en SQLite. Une issue incertaine ne déclenche
aucun nouvel appel SSH.

```powershell
python -m unittest tests.test_ood_gpu_workflow -v
```

Ces tests utilisent un transport SSH simulé, y compris pour les retours
mal formés et les dépassements de délai. La première calibration sur la
machine préparée demandera l'autorisation affichée par le workflow.

## Démarrer et lire la décision en attente

```powershell
$workflow = "runs_control/recherche-relance"
python -m scripts.research_workflow --dir $workflow start --id recherche-relance --brief examples/research_brief_retry.md --provider openai --model MODELE_DE_PLANIFICATION
python -m scripts.research_workflow --dir $workflow status
```

Remplacer `MODELE_DE_PLANIFICATION` par un identifiant auquel le compte a accès.
OpenAI et Anthropic sont pris en charge pour la planification. `start` prépare
le premier appel et ses bornes sans joindre le fournisseur.

Pour utiliser GPT-6.2 Astra comme modèle du scientifique, choisir
`--provider openai --model gpt-6.2-astra`. Le provider demande un niveau de
raisonnement `low`, visible dans les demandes d'autorisation de ces appels.
Le protocole et l'exécuteur déterminent de leur côté quels modèles sont étudiés.

### Autoriser une mission OOD en une fois

Sur une nouvelle mission `ood_gpu` ou `ood_replay`, préparer l'autorisation
juste après `start`. Le mode autonome réutilise les contrôles, les décisions
et le journal du mode guidé. Il vérifie toujours une décision au moment où
elle s'applique et l'enregistre au nom de la délégation.

```powershell
python -m scripts.research_workflow --dir $workflow autonomy-plan
python -m scripts.research_workflow --dir $workflow autonomy --actor operateur --target-sha256 EMPREINTE_AFFICHEE
```

Après examen, saisir `APPROVE` une seule fois. Le plan lie le brief, le
fournisseur, le modèle, les requêtes bibliographiques que l'agent pourra
former à partir du brief, le lot expérimental et l'export local du manuscrit.
Pour `ood_gpu`, il affiche également le serveur, la carte, la révision de Qwen
et les quatre entrées fixées. Le budget par défaut couvre quatre appels LLM,
huit requêtes OpenAlex avec cinq résultats chacune, deux révisions et un lot.
Les options `--llm-calls`, `--openalex-queries` et `--revisions` permettent de
préparer un autre budget avant son approbation. Utiliser les mêmes options pour
afficher et autoriser le plan.

L'agent peut raccourcir une recherche trop étroite ou demander une nouvelle
synthèse quand une citation ne correspond pas au catalogue. Chaque tentative
consomme le budget au moment de sa réservation. Le workflow s'arrête si le
budget est épuisé, si le périmètre change, si la calibration GPU de référence
échoue ou si une opération reste incertaine. Après une interruption, la commande
`resume-autonomy` relit le journal et reprend dans les limites du même plan.
Un job réservé sans reçu complet demande toujours une réconciliation humaine.

Les validations internes portent sur la structure, les preuves disponibles et
le périmètre d'exécution. Le manuscrit exporté est un rapport de calibration.
Les autres charges de travail utilisent le mode guidé.

### Suivre la mission pendant son exécution

Dans un deuxième terminal, depuis la racine du dépôt, lancer `watch` avec le
même dossier que pour `autonomy`.

```powershell
python -m scripts.research_workflow --dir runs_control/ood-gpu-calibration watch
```

Le suivi relit le journal SQLite chaque seconde. Il affiche les étapes réservées
et terminées, les incidents, les révisions et le chemin de l'export. Pour un appel
long, l'étape reste affichée comme réservée jusqu'à l'enregistrement du résultat.
`Ctrl+C` ferme l'observateur et la mission continue dans son terminal. Utiliser
`watch --once` pour consulter les étapes déjà enregistrées ou `status` pour lire
l'état détaillé et la décision en attente. Ces commandes lisent le journal local.

### Page locale et archives PDF ACL

```powershell
python -m scripts.research_workflow --dir runs_control/ood-gpu-calibration dashboard
```

Ouvrir `http://127.0.0.1:8765/`. La page met à jour le journal pendant
l'exécution. Sélectionner un jalon pour consulter son état, puis utiliser
« Revenir au direct » pour suivre le dernier. Les brouillons révisés et le
manuscrit approuvé ont chacun leur PDF ACL. Le tableau de bord peut aussi
rouvrir une mission terminée et reconstituer ses éditions à partir du journal.

Le journal SQLite conserve les étapes. Les manuscrits sont vérifiés avant de
générer leur LaTeX, à partir des affirmations, des références et des agrégats
déjà enregistrés. Chaque édition est conservée dans `dashboard/pdfs` dans le
dossier de la mission, avec son source LaTeX, le style ACL et leurs SHA-256.
Le PDF porte la mention « Brouillon de travail » ou « Version approuvée ».
Une édition archivée et vérifiée est réutilisée au redémarrage.

La page et la compilation utilisent la bibliothèque standard Python et
`pdflatex`. Le style [officiel ACL](https://github.com/acl-org/acl-style-files)
fixe la mise en page. Le poste Windows préparé utilise TinyTeX installé sous
`runs_control/acl-tools/tinytex` et le style `runs_control/acl-tools/acl.sty`.
Sur un autre poste, `--pdflatex` et `--acl-style` permettent d'indiquer des
chemins locaux. `dashboard --archive-only` compile les éditions enregistrées
et quitte. Le tableau de bord est accessible sur la boucle locale uniquement.

### Traces des agents et métriques W&B

Les agents déclarent une version de leurs consignes dans leur constructeur
`AgentRunner`. Incrémenter `prompt_version` quand le contrat de rôle change.
L'identifiant est le nom du rôle. Le SHA-256 couvre les consignes effectivement
transmises, y compris les règles de mémoire ou le schéma ajouté au texte.
Le journal associe l'appel à sa réservation, puis à son résultat technique
et au verdict du workflow. Les appels historiques antérieurs à cette
instrumentation peuvent être absents de l'export.

```powershell
python -m scripts.research_workflow --dir runs_control/ood-gpu-calibration telemetry
python -m pip install -r requirements-observability.txt
python -m scripts.research_workflow --dir runs_control/ood-gpu-calibration telemetry --wandb-offline
```

La commande lit les journaux existants et écrit un `metrics.json` sous
`telemetry/`. Pour les pilotes OOD, les scores et les mesures sont reconstruits
à partir des reçus vérifiés. L'option W&B crée un run hors ligne avec les étapes,
les latences des agents et, lorsque le reçu les fournit, les tokens, durées et
métriques GPU. L'export est limité aux champs techniques déclarés. Le brief,
les textes de consigne, les réponses et les secrets restent locaux. Le mode
hors ligne désactive aussi la collecte du code, de la console et des
métadonnées système par W&B.

Le workflow de code V7.4 expose séparément les durées, statuts, codes de sortie
et empreintes des tentatives de sandbox. Il peut également exporter les traces
de l'agent qui propose ou corrige le code. Utiliser un thread déjà enregistré.

```powershell
python -m scripts.experiment_workflow telemetry --thread MON_THREAD --wandb-offline
```

Pour envoyer les traces techniques de la mission vers un projet Weave,
installer `requirements-weave.txt` et configurer les identifiants du service
dans l'environnement ou via la connexion W&B. `telemetry --weave-project
EQUIPE/PROJET` affiche le projet et les traces avant la confirmation `APPROVE`.
Ce transfert porte sur les appels instrumentés et terminés. La capture
automatique des appels et du code est désactivée. SQLite conserve les preuves
utilisées par l'analyse et par la reprise du workflow.

### Valider chaque étape en mode guidé

```powershell
python -m scripts.research_workflow --dir $workflow guide --actor operateur
```

Le mode guidé affiche chaque demande dans le terminal, puis attend
`APPROVE` ou `STOP`. Il propose les actions et passe lui-même à la demande
suivante après ton accord. `STOP` laisse la décision en attente. Relancer
`guide` reprend au même endroit.

| Demande | Moment |
|---|---|
| Autorisation | Avant un appel modèle, la découverte OpenAlex, la calibration, la mesure ou une campagne Docker locale. |
| Validation | Après la proposition de mission, la synthèse, le protocole et le manuscrit. |
| Retour en arrière | Quand un contrôle détecte un défaut et que le workflow dispose d'une correction concrète. |
| Intervention | Quand il manque un exécuteur, que la calibration a échoué sans correction connue, ou qu'un appel demeure incertain. |

Pour traiter une demande isolément ou automatiser l'interface, `status` expose
la demande dans `pending`. On peut ensuite approuver son empreinte :

```powershell
python -m scripts.research_workflow --dir $workflow approve --target-sha256 SHA_ACTION --actor operateur
```

La commande demande aussi `APPROVE`. Les clés restent dans l'environnement ou
le fichier `.env` local. La mission et la littérature
approuvées sont exportées dans `versions/vNN`. Les essais et le manuscrit
sont dans le dossier `pilot-NN` de leur protocole. Aucun transfert manuel
entre étapes n'est nécessaire.

Le brief peut déclarer `## Exécuteur` avec `retry_catalog`. L'opérateur peut
aussi choisir cette valeur à `start` avec `--workload`. Le protocole montre
la question demandée et la question effectivement mesurée. Ce profil fixe
utilise le runner V9, un catalogue compatible et son modèle expérimental,
distinct du modèle de planification. Un corpus compatible peut être donné
avec `--corpus`. Pour un brief sans exécuteur, la chaîne prépare un plan V8.3
et s'arrête après sa validation en signalant le raccord qui manque.

## Tester le moteur avec des valeurs synthétiques

Choisir `--workload synthetic.mean` à la place de `retry_catalog`, ou déclarer
`synthetic.mean` dans la section `## Exécuteur` du brief. Si le brief déclare
déjà un autre exécuteur, utiliser un nouveau brief ou revenir au protocole
avec `revise --stage protocol --workload synthetic.mean`. L'image Docker locale
`agentic-lab-sandbox:v7.2` doit déjà être disponible.

```powershell
$workflow = "runs_control/pilote-synthetique"
python -m scripts.research_workflow --dir $workflow start --id pilote-synthetique --brief tests/fixtures/research_brief.md --provider openai --model MODELE_DE_PLANIFICATION --workload synthetic.mean
python -m scripts.research_workflow --dir $workflow guide --actor operateur
```

Le protocole reprend les deux listes et les limites de ressources de
[`examples/v846_protocol.json`](../examples/v846_protocol.json). Il affiche
aussi la question demandée et le calcul effectivement réalisé. Après la revue
du protocole, `guide` demande une autorisation Docker séparée qui montre les
valeurs, la graine, les ressources et l'empreinte de chaque job. Le moteur
réutilise son programme autorisé et vérifie les métriques d'après les entrées,
même après une sortie normale du conteneur.

Le parcours se termine sur un rapport d'intégration `synthetic-NN/report.json`.
Le rapport lie le protocole et son autorisation aux jobs et aux reçus SQLite.
`status` affiche son chemin et vérifie son empreinte. En cas d'interruption,
`recover` peut rattacher les résultats si tous les reçus sont déjà vérifiables.
Les moyennes sont des observations sur les listes fictives du pilote
(`scientific_evidence: false`). La préparation en amont conserve les
autorisations habituelles pour les appels au planificateur et à OpenAlex.

## Plan d'action GPU pour le pilote de relance

`retry_gpu` soumet un seul lot de 32 essais. Il croise les deux politiques
(aucune relance ou une relance) et les deux conditions de lecture (normale
ou avec panne transitoire) sur chacune des huit tâches du corpus. Le modèle
étudié est Qwen3-4B-Instruct-2507 à révision fixe. Le fournisseur du
planificateur reste configurable séparément. Le pilote couvre cette
comparaison précise.

Le [compilateur du plan](../execution/retry_gpu.py) transforme le calendrier
apparié en job après revue du protocole. Les entrées sont un corpus explicite,
dont chaque consigne contient la référence du produit attendu, et une
configuration SSH qui désigne le fichier `retry_gpu_worker.py`, le
Python distant, le snapshot Qwen et l'index CUDA. Le [programme
d'exécution](../execution/retry_gpu_worker.py) charge le snapshot localement.
Il peut lire une valeur du catalogue pour la tâche courante. Il applique
l'éventuelle relance aux mêmes arguments et transmet au modèle le résultat
de la lecture. Les réponses du modèle sont traitées comme du JSON borné.

```powershell
$workflow = "runs_control/pilote-relance-gpu"
python -m scripts.research_workflow --dir $workflow start --id pilote-relance-gpu --brief CHEMIN_BRIEF --provider openai --model MODELE_DE_PLANIFICATION --workload retry_gpu --corpus examples/v9_retry_pilot.json --gpu-config CHEMIN_CONFIG_GPU
python -m scripts.research_workflow --dir $workflow guide --actor operateur
```

Après approbation du protocole, `pilot-NN/action-plan.json` contient les
essais, la cible SSH, les empreintes et le budget. L'autorisation de ce plan
couvre le lot entier, avec une carte GPU, 64 générations au plus et un délai
distant de 15 minutes. Il faut préalablement déployer le fichier worker sur
l'hôte approuvé et y disposer du snapshot local. Une modification du protocole,
du corpus ou de la configuration demande une révision et une nouvelle
autorisation. Après une coupure, `recover` rattache un reçu déjà enregistré.
Une soumission d'issue inconnue reste en attente de réconciliation.

Le workflow écrit `pilot-NN/report.json` avec les scores, les écarts appariés,
les tentatives d'outil et la chaîne d'empreintes jusqu'au reçu SQLite.
`status` affiche son chemin. Le critique et le rédacteur relisent le reçu
original pour produire le brouillon et indexer les observations dans la mémoire.
Le tableau de bord compile le brouillon au format ACL et archive son édition.
Le rapport est descriptif. Le parcours `guide` affiche une décision distincte
pour approuver et exporter le manuscrit final.

## Exécuter un protocole inédit depuis un brief

Pour une mission lancée avec un brief seul, après la revue bibliographique
le sélecteur compare la question approuvée aux ressources disponibles.
Le catalogue initial couvre les paires de documents du [pilote de
classement](../examples/v9_new_protocol_brief.md) et les additions avec deux
formats de consigne si une cible GPU est configurée. Le planificateur reçoit
le nombre de tâches, les conditions vérifiables et la métrique du corpus
sélectionné. Le journal conserve la justification, la source et l'empreinte
du corpus. Les réponses de référence restent dans le workflow.

```powershell
$workflow = "runs_control/nouveau-protocole"
python -m scripts.research_workflow --dir $workflow start --id classement --brief examples/v9_new_protocol_brief.md --provider openai --model MODELE_DE_PILOTAGE
python -m scripts.research_workflow --dir $workflow autonomy-plan
python -m scripts.research_workflow --dir $workflow autonomy --actor operateur --target-sha256 SHA_PLAN
```

La délégation couvre le corpus, la capacité d'exécution, la recherche dans
OpenAlex et le nombre d'appels au fournisseur. Le budget par défaut est de
six appels quand la sélection est automatique. Le brouillon attend une
décision éditoriale. `guide` permet de valider les étapes séparément. Si le
catalogue ne couvre pas la question, le workflow passe à
`capability_resolution` et conserve la justification du sélecteur.

1. Le résolveur examine la question approuvée et le manque signalé par le catalogue.
   Il dispose d'un appel au fournisseur, décompté du budget de la mission, et
   d'une seule tentative dans la délégation. Le mode `guide` affiche son autorisation.
2. Pour un pilote numérique compatible, le résolveur choisit deux règles et
   une cible explicite parmi minimum, maximum, somme et différence absolue.
   Une recette versionnée produit six paires mesurées et trois exemples de
   contrôle par règle. Elle calcule les références indépendamment du code
   proposé par l'agent. La recette s'exécute sur CPU local sans accès réseau.
3. Le workflow enregistre le diagnostic, la ressource et le corpus dans
   `versions/vXX/` sous `capability-resolution.json`, `resolved-resource.json`
   et `resolved-evaluation.json`. Il reconstruit le corpus pour vérifier son
   empreinte et reprend au protocole. L'agent de code doit réussir les exemples
   de contrôle avant l'exécution des mesures. Le résultat reste un pilote
   synthétique descriptif.

Si la capacité sort du périmètre, si un accès ou des données sont requis ou
si le protocole manque d'une définition mesurable, le workflow conserve le
diagnostic et indique le prérequis. Le suivi `progress` affiche l'étape de
résolution. `--evaluation` au démarrage ou
`revise --stage protocol --evaluation CHEMIN` permet toujours de fournir
un corpus indépendant compatible.

Le fichier d'évaluation distingue les cas mesurés (`cases`) des exemples
de contrôle (`witnesses`). Un témoin doit couvrir chaque cellule du
protocole. Seuls les témoins et les noms des entrées sont transmis à l'agent
de code. Une implémentation qui échoue sur un témoin peut être révisée dans
le budget. La version précédente et la cause du retour restent dans le
journal.

Les capacités disponibles utilisent le même plan d'action.

| Capacité | Programme proposé | Mesure |
|---|---|---|
| `cpu.integer_exact.v1` | Fonction Python réduite à une expression numérique bornée. | Prédiction sur CPU et comparaison avec un entier de référence. Jusqu'à 240 évaluations. |
| `gpu.integer_prompt.v1` | Expression Python qui choisit une variante parmi les prompts approuvés de la tâche. | Inférence Qwen3-4B à révision fixe sur une carte CUDA. Jusqu'à 32 générations par lot. |

Le profil GPU attend deux à quatre champs de texte dans la liste `prompts`
de chaque cas et une configuration `--gpu-config` pointant vers
`protocol_gpu_worker.py`. Chaque témoin GPU attend l'indice de variante
commençant à zéro. La réponse de référence des cas mesurés reste un entier
à comparer à la sortie du modèle. Installe le nouveau worker sur l'hôte
approuvé et configure sa cible SSH une seule fois. Le modèle et sa révision
sont indiqués dans le plan. La configuration peut désigner un cache avec
`model_cache` au lieu d'un `snapshot_path` explicite.

```json
{
  "ssh_host": "mon-hote-gpu",
  "python_path": "/home/operator/lab-gpu/.venv/bin/python",
  "worker_path": "/home/operator/lab-gpu/protocol_gpu_worker.py",
  "model_cache": "/home/operator/lab-gpu/models",
  "gpu_index": 0,
  "allow_model_download": true
}
```

Avec `allow_model_download: true`, le plan peut autoriser une tentative
de préparation de la révision Qwen publiée sur Hugging Face, plafonnée
à environ 8,06 Go et 30 minutes. Le worker réutilise les fichiers dont les
empreintes concordent et récupère les fichiers manquants. Il vérifie chaque
taille et chaque empreinte avant d'installer les fichiers. Les fichiers
existants incompatibles exigent un autre cache. Le reçu
`pilot-NN/model-preparation.json` indique si le snapshot a été installé ou
réutilisé. La mesure charge ensuite le snapshot en mode hors ligne. La
valeur par défaut de `allow_model_download` est `false`.

Le workflow interprète l'expression de l'agent avec une grammaire bornée.
L'accès au réseau, au système de fichiers, aux imports et aux appels de
fonctions reste hors de cette grammaire. Le worker GPU reçoit les prompts
retenus et des paramètres validés. Le plan publié conserve la relation entre
protocole, programme et jobs. À partir des reçus SQLite, l'analyse calcule
les taux par condition et les contrastes appariés. Le tableau de bord peut
compiler le brouillon ACL. Tant que l'issue d'une soumission est inconnue,
`recover` peut uniquement rattacher des reçus complets.

## Pilote multi-checkpoint sur des tâches avec outil

Le [brief de démonstration](../examples/v9_agentic_ood_brief.md) peut utiliser
le corpus `agentic_catalog_ood_v1` lorsque la cible déclarée utilise
`agentic_gpu_worker.py`. Le sélecteur transmet son choix au planificateur.
Le protocole conserve la question, la provenance du corpus, les checkpoints
fixés, les modalités et le budget. Un protocole qui modifie cette grille ne
peut pas être compilé par ce profil.

La grille comprend **24 cellules et 144 essais**. Elle croise trois modèles
(Qwen3-1.7B, Qwen3-4B-Instruct-2507 et OLMoE-1B-7B-0125-Instruct), deux
politiques (aucune relance ou une relance) et quatre conditions (contrôle,
reformulation, panne transitoire et distraction). Trois tâches de lecture
et deux ordres fixes sont appliqués à chaque cellule. Chaque essai utilise
exactement deux générations. La première demande une lecture et la seconde
propose une réponse à partir du résultat. Seule la lecture de la référence
prévue peut réussir. Le résultat final doit être un entier exact au format
JSON et s'appuyer sur une lecture réussie.

Les [révisions de Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B/tree/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e),
[Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/tree/cdbee75f17c01a7cc42f958dc650907174af0554)
et [OLMoE](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/tree/b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e)
sont épinglées. Les contrastes entre checkpoints combinent des différences
d'entraînement, de taille et parfois d'architecture. Les écarts OOD sont
appariés au contrôle pour les mêmes tâches et le rapport montre la
sensibilité en retirant une tâche à la fois. Le corpus de trois tâches
fournit une mesure descriptive à cette étape.

### Configurer la cible agentique

Déploie `execution/agentic_gpu_worker.py` et
`execution/protocol_gpu_worker.py` ensemble dans le même dossier sur
l'hôte GPU. Le premier exécute le benchmark. Le second fournit ses
fonctions de validation des fichiers et des destinations de téléchargement.
L'empreinte du worker couvre les deux fichiers.

```json
{
  "ssh_host": "mon-hote-gpu",
  "python_path": "/home/operator/lab-gpu/.venv/bin/python",
  "worker_path": "/home/operator/lab-gpu/agentic_gpu_worker.py",
  "snapshot_path": "/home/operator/models/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554",
  "model_cache": "/home/operator/lab-gpu/models",
  "gpu_index": 0,
  "allow_model_download": true
}
```

`snapshot_path` désigne le Qwen3-4B et permet de réutiliser le cache déjà
validé. Les chemins des deux autres modèles sont dérivés de `model_cache`.
Le plan de délégation affiche les révisions, les octets maximum et les
plafonds de temps. Tous les snapshots doivent passer la vérification avant
le premier job. Les trois lots utilisent la même carte, l'un après l'autre.
Une soumission d'issue inconnue interrompt le lot et conserve les reçus
déjà terminés.

```powershell
$workflow = "runs_control/pilote-agentic-ood"
python -m scripts.research_workflow --dir $workflow start --id agentic-ood --brief examples/v9_agentic_ood_brief.md --provider openai --model MODELE_DE_PILOTAGE --gpu-config CHEMIN_CONFIG_AGENTIQUE
python -m scripts.research_workflow --dir $workflow autonomy-plan
python -m scripts.research_workflow --dir $workflow autonomy --actor operateur --target-sha256 SHA_PLAN
```

Le parcours automatisé assemble `study.json`, `action-plan.json`, les
reçus de préparation et le journal SQLite. Le rapport et le brouillon ACL
proviennent des reçus GPU complets. Pour une vérification locale avec
transport et générations simulés, lancer
`python -m unittest tests.test_agentic_gpu_workflow -v`.

## Corriger et reprendre

Quand la couverture est insuffisante, le workflow essaie de proposer des
requêtes plus courtes issues de la mission. Quand un extrait cité ne figure
pas dans le catalogue, il propose de reprendre la synthèse en conservant les
notices. La correction affiche sa raison et sa cible. Son approbation est
locale. Le prochain appel éventuel demande une autorisation séparée.

Tu peux aussi revenir volontairement en arrière, par exemple si une hypothèse
est discutable malgré un format valide. L'empreinte attendue par `revise` est
`checkpoint_sha256`, affichée par `status`.

```powershell
python -m scripts.research_workflow --dir $workflow revise --stage catalog --reason "Élargir le vocabulaire de recherche" --query "tool failure recovery in language model agents" --target-sha256 SHA_CHECKPOINT --actor operateur
python -m scripts.research_workflow --dir $workflow history
```

| Retour | Périmètre recalculé |
|---|---|
| `mission` | Toute la chaîne. Accepte aussi un brief corrigé avec `--brief`. |
| `catalog` | La recherche bibliographique et ses dépendances. `--query` peut être répété. |
| `literature` | La synthèse et ses dépendances. Le catalogue déjà collecté est réutilisé. |
| `protocol` | Le plan et les expériences associées. Un nouvel exécuteur ou un corpus compatible peut être précisé. |
| `implementation` | Un nouveau programme pour le même protocole approuvé. L'essai de contrôle précède la mesure. |
| `draft` | Une nouvelle version du brouillon à partir des mêmes mesures vérifiées. |

La raison est consignée. Elle est transmise au planificateur ou à l'analyste
si l'étape implique une nouvelle proposition modèle. Chaque retour crée une
version et retire les descendants de l'état actif. Les anciennes décisions,
pièces et mesures restent accessibles, et leurs approbations ne s'appliquent
plus à la version active. Le workflow limite le parcours à quatre retours.

Après interruption, `recover --target-sha256 SHA_CHECKPOINT --actor operateur`
peut refaire une génération de brouillon locale ou rattacher une phase déjà
terminée dont le reçu est vérifiable. Une opération réservée à l'issue inconnue
exige une réconciliation humaine. Les appels distants ne sont jamais répétés
automatiquement dans ce cas.

Le test de bout en bout utilise des fournisseurs simulés et couvre le brief,
deux retours en amont, une reprise, l'expérience et l'approbation du manuscrit.
Les références du manuscrit sont rattachées à la revue approuvée. La vérification
des articles porte sur les notices et extraits de résumés.
