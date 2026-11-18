# V9 — Protocole pilote pour une relance d'outil

Le plan d'exécution est prêt pour la validation de la calibration.

## Question

À modèle et tâche identiques, est-ce qu'une relance automatique après une
erreur transitoire d'un outil de lecture améliore le taux de réponses exactes ?
L'intervention est une règle de l'orchestrateur. Le modèle reçoit les mêmes
consignes dans les deux conditions.

| Profil de l'outil | Sans relance | Une relance |
|---|---|---|
| Fonctionnement normal | Une lecture | Une lecture |
| Erreur transitoire injectée au premier appel valide | L'erreur est renvoyée au modèle | Les mêmes arguments sont rejoués une seule fois |

L'outil lit uniquement des champs autorisés dans un catalogue local. Les
arguments invalides ne déclenchent aucune relance. Les lectures sont sans effet
de bord et l'erreur injectée se produit avant la lecture. Le catalogue est
fictif. Les appels et réponses observés lors du futur pilote seront ceux du
modèle choisi, sur ces tâches contrôlées.

## Calendrier des essais

- Huit tâches distinctes, réparties entre demandes de stock et de zone. Chaque
  tâche est testée dans les quatre conditions. L'unité appariée est la tâche,
  ce qui donne huit paires principales en profil transitoire.
- Une session neuve à chaque essai. Même modèle exact, même fournisseur ou
  runtime, même définition d'outil, mêmes consignes et mêmes limites. Le modèle
  ne reçoit ni le nom de la condition ni le catalogue complet. La valeur
  attendue sert uniquement à la notation hors de la conversation.
- L'ordre des tâches est déterminé par une graine fixe. Les ordres des profils
  et des politiques sont équilibrés entre tâches. Les appels autorisés sont un
  appel de décision et, si un résultat d'outil est à transmettre, un appel de
  réponse finale. Le plan complet prévoit 32 essais et au plus 64 appels modèle.
- La calibration utilise deux autres références, chacune testée avec les
  deux politiques. La première sert au profil normal, la seconde au profil
  transitoire. Elle passe si les quatre appels d'outil sont valides, si les
  lectures abouties donnent la valeur exacte et si le témoin en erreur
  produit une abstention. La phase mesurée a sa propre approbation.

## Mesure et critères

Le modèle reçoit un format de réponse unique `{"answer": "valeur"}` et peut
rendre `{"answer": null}` en cas d'abstention. La mesure principale est la
réponse finale exacte selon le format, comparée à la valeur du catalogue
sans jugement de LLM. L'analyse rapporte les succès dans chaque bras en profil
transitoire, les paires où seule la relance réussit ou échoue et la différence
de taux appariée. Le profil normal sert de témoin. Les mesures secondaires
sont le recours à l'outil, les tentatives physiques, les abstentions, les
latences et l'usage de tokens si le backend le fournit.

Une réponse fausse, hors format ou absente après un échange complet compte
comme un échec. Une panne du fournisseur ou une issue inconnue arrête la
série et marque le plan incomplet. Les traces partielles sont conservées.
Une reprise exige une décision humaine sur les essais concernés.

Le pilote est exploitable si toutes les conditions prévues ont une trace
complète, si l'analyse est recalculable à partir des reçus, si chaque essai
respecte le budget et si le témoin permet d'examiner un éventuel écart en
fonctionnement normal. Le pilote reste descriptif. Huit tâches et un modèle
servent à juger le montage et à dimensionner une étude plus large.

## Modèle et budget à valider

Le modèle étudié est le snapshot `gpt-5.4-mini-2026-03-17` via l'API
Responses. La version datée fixe l'identité du modèle et prend en charge les
appels d'outils. Ce réglage est propre au pilote. Le runner utilise le tarif
standard, sans outil hébergé, avec 1 024 tokens d'entrée vérifiés avant
génération et 256 tokens de sortie maximum par génération. Le délai est de
30 secondes par requête et le SDK a zéro nouvelle tentative automatique.

| Phase | Essais | Générations maximum | Requêtes de comptage maximum | Enveloppe de génération |
|---|---:|---:|---:|---:|
| Calibration | 4 | 8 | 8 | 0,01536 USD |
| Mesure | 32 | 64 | 64 | 0,12288 USD |
| Total | 36 | 72 | 72 | 0,13824 USD |

L'enveloppe utilise les tarifs standard hors cache de 0,75 USD par million
de tokens d'entrée et 4,50 USD par million de sortie, vérifiés le
18 novembre 2026. Elle estime les générations et reste distincte du montant
facturé par le fournisseur. Le journal conserve le coût estimé des reçus
disponibles. Une génération dont l'issue est inconnue peut avoir été facturée.
[Fiche modèle et tarifs](https://developers.openai.com/api/docs/models/gpt-5.4-mini)

## Vérification et lancement

Depuis la racine, prévisualiser le fichier public de tâches et vérifier les
règles de l'outil, sans appel modèle :

```powershell
python -m scripts.demo_v9_pilot preview
python -m unittest tests.test_v9_retry_pilot tests.test_v9_retry_pilot_runner -v
```

La prévisualisation donne l'empreinte du plan, du corpus et du code de la
boucle d'exécution. Le journal SQLite conserve les approbations, les intentions
d'appel avant le réseau, les reçus du fournisseur, les événements de lecture
et le score par essai. Le statut se lit hors ligne. Une phase interrompue
reste consultable et le runner refuse de la rejouer sur le même journal.

```powershell
$journal = "runs_control/v9_retry_pilot.sqlite3"
python -m scripts.demo_v9_pilot approval --journal $journal --phase calibration
python -m scripts.demo_v9_pilot execute --journal $journal --phase calibration --actor operateur --target-sha256 SHA_APPROBATION
python -m scripts.demo_v9_pilot status --journal $journal
```

Remplacer `SHA_APPROBATION` par celui affiché par `approval`. La commande
`execute` demande aussi de saisir `APPROVE` avant tout accès réseau. Si la
calibration passe, demander la nouvelle empreinte avec `--phase measurement`.
Elle lie la phase mesurée aux reçus de calibration. Les clés restent dans
l'environnement ou le fichier `.env` local.

Après exécution, le rapport reliera chaque mesure à la tâche, à la politique,
aux événements d'outil et à la réponse. La critique, la mémoire et le manuscrit
utiliseront un contrat de résultats V9 qui conserve la portée du pilote. Les
formats de données synthétiques V8.5 à V8.9 restent réservés à leurs démos.
