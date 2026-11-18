# V8.9 — Parcours de recherche intégré

Le parcours part d'un dossier V8.3 approuvé qui contient la mission V8.1,
la littérature V8.2, le protocole et sa matrice. Le pilote demande une liaison
explicite entre chaque cellule de la matrice et un jeu de nombres synthétiques.
Le plan affiche la cellule, ses facteurs, le job V8.4 et son empreinte.
Il autorise uniquement le workload existant `synthetic.mean`, pour deux à
quatre cellules. Cette liaison valide le passage de la conception aux jobs
dans le laboratoire. Les moyennes calculées décrivent les nombres fournis.

| Composant | Rôle |
|---|---|
| LangGraph | Relie le plan, la décision d'exécution, les résultats, la revue, la mémoire et le manuscrit. Chaque décision est liée à un SHA-256. |
| SQLite | Conserve résultats, réservations, revues, décisions et mémoire locale entre les processus. Une soumission incertaine bloque la suite pour examen humain. |
| PostgreSQL | Conserve les checkpoints LangGraph pour reprendre le parcours entre commandes. |
| Transport arithmétique | Calcule les moyennes en mémoire et émet des reçus marqués simulés dans le format V8.4. Le contrôleur V8.5 revérifie les mesures depuis le journal. |

La revue V8.6 utilise V8.7 pour consulter les sources historiques et conserver
sa revue et ses décisions. Le dossier d'origine est référencé par la revue
et le manuscrit. Le plan V8.9 porte la liaison avec le pilote. Le contrat V8.6
continue à traiter les révisions comme des propositions prospectives. Leur
validation exige une décision distincte et laisse les jobs du pilote intacts.

Depuis la racine, sans serveur ni clé API :

```powershell
python -m scripts.demo_v89 plan --input examples/v89_workflow.json
python -m scripts.demo_v89 --root runs_control/v89_demo --run-id v89-demo offline --input examples/v89_workflow.json
```

Le parcours demande deux confirmations `APPROVE`, pour les jobs simulés puis
pour le manuscrit. Cette démonstration choisit de poursuivre sans révision.
Utiliser le mode persistant pour déposer et relire une révision.

Avec un PostgreSQL déjà autorisé et `V89_CHECKPOINT_DATABASE_URL` configurée :

```powershell
python -m scripts.demo_v89 --root runs_control/v89_persistent --run-id v89-demo start --input examples/v89_workflow.json
python -m scripts.demo_v89 --root runs_control/v89_persistent --run-id v89-demo status
python -m scripts.demo_v89 --root runs_control/v89_persistent --run-id v89-demo resume --decision approve --target-sha256 <SHA> --actor operateur
```

Consulter le statut après chaque reprise et recopier l'empreinte affichée.
Le critique attend `resume --decision skip` ou `--decision propose --draft <JSON>`
au format V8.6. Une proposition validée peut être approuvée, refusée ou renvoyée
pour modifications motivées. Utiliser un identifiant inédit pour chaque parcours
dans une même base de checkpoints. Pour partager une mémoire SQLite existante
entre plusieurs parcours, reprendre les mêmes options globales
`--memory-db <base> --project <projet>` à chaque commande concernée.
`recover` sert à retenter un nœud interrompu sans décision en attente.
Le journal bloque toujours une nouvelle soumission dont l'état est incertain.

Les réglages facultatifs `critic_settings` et `writer_settings` dans le JSON
activent les requêtes OpenAI ou Anthropic en mode persistant. Chaque requête
affiche son contenu, son fournisseur et ses bornes avant une décision dédiée.
L'approbation des jobs ou d'un document reste distincte de l'autorisation API.

Après approbation du manuscrit, `output/` contient le plan, le rapport
d'analyse, la revue, le manuscrit et `evaluation.json`. Ce dernier compte
les jobs, les requêtes LLM enregistrées, les décisions conservées, les souvenirs
retrouvés et les affirmations avec références. Il indique zéro coût API pour
le parcours hors ligne. Les reçus actuels ne collectent pas la facturation
des vrais appels.

Les scénarios de test couvrent le succès, le refus, l'altération des sources
ou de la figure, la reprise sur PostgreSQL, le rappel d'une décision par
rapport à un projet témoin vide et l'arrêt sur soumission incertaine. Le pilote
valide le fonctionnement du parcours. L'effet de la mémoire sur la qualité
des décisions de modèles réels, les workloads scientifiques et Jean Zay
sont à évaluer lors des travaux suivants.
