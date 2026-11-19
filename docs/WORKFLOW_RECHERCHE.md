# Du brief au manuscrit

Le point d'entrée `scripts.research_workflow` prend un brief Markdown une seule
fois. Le workflow reprend la mission, la découverte OpenAlex et la synthèse
V8.1–V8.3. Il prépare ensuite un protocole et vérifie la présence d'un exécuteur
compatible. Un journal SQLite local conserve chaque version, les approbations
et la réservation qui précède chaque effet externe. LangGraph reste utilisé
par les composants existants, dont la critique du pilote.

## Démarrer et lire la décision en attente

```powershell
$workflow = "runs_control/recherche-relance"
python -m scripts.research_workflow --dir $workflow start --id recherche-relance --brief examples/research_brief_retry.md --provider openai --model MODELE_DE_PLANIFICATION
python -m scripts.research_workflow --dir $workflow status
```

Remplacer `MODELE_DE_PLANIFICATION` par un identifiant auquel le compte a accès.
OpenAI et Anthropic sont pris en charge pour la planification. `start` prépare
le premier appel et ses bornes sans joindre le fournisseur.

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
