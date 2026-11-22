# V8.8 — Scientific Paper Writer

Le rédacteur transforme une revue V8.6 terminée en manuscrit pédagogique.
Il rouvre les sources d'origine et reconstruit les calculs, la revue et, si
elle existe, la révision approuvée. Pour chaque affirmation, le JSON indique
l'artefact, son SHA-256, le chemin dans cet artefact et le SHA-256 de la valeur.
Un éditeur ou un lecteur peut donc vérifier le chiffre ou le texte cité.

Python construit les sections, la bibliographie et la figure de façon
déterministe. La bibliographie reprend les notices effectivement citées et
les extraits contrôlés dans les résumés V8.2. La figure SVG affiche les
moyennes issues de l'analyse V8.5, avec un axe centré sur zéro et les valeurs
de chaque barre. Les hypothèses, remarques proposées par le critique, résultats
synthétiques et révisions prospectives conservent leur statut dans le texte.

L'assistance OpenAI ou Anthropic est facultative. Le modèle propose un ordre
pour les affirmations existantes, à partir de leurs identifiants. Python exige
une permutation complète, garde l'ordre des sections et produit tous les textes
et chiffres. L'appel réutilise le journal de V8.6, avec une approbation exacte,
une seule réservation par revue, un plafond de sortie et aucun nouvel essai
automatique. Si un contexte mémoire V8.7 a été consulté lors de la revue, son
empreinte est conservée. Un appel éditorial demande aussi sa revalidation.

Depuis la racine du dépôt, avec une revue V8.6 déjà enregistrée :

```powershell
python -m scripts.tutorials.scientific_manuscript --store runs_control/v86_reviews.sqlite3 --review v86-demo-001 preview
python -m scripts.tutorials.scientific_manuscript --store runs_control/v86_reviews.sqlite3 --review v86-demo-001 decide --decision approve --actor relecteur --target-sha256 <SHA>
python -m scripts.tutorials.scientific_manuscript --store runs_control/v86_reviews.sqlite3 --review v86-demo-001 export --target-sha256 <SHA> --out-dir runs_control/paper-demo
python -m scripts.tutorials.scientific_manuscript --store runs_control/v86_reviews.sqlite3 --review v86-demo-001 verify --target-sha256 <SHA> --out-dir runs_control/paper-demo
```

Remplacer `<SHA>` par l'empreinte affichée par `preview`. `decide` demande une
confirmation sur la version affichée. Pour demander une réorganisation assistée,
`assist --provider openai|anthropic --model <modèle> --actor <nom>` présente
d'abord la requête pour approbation. Utiliser ensuite `preview --assisted` et
`decide --assisted` pour relire et décider sur la version assistée.

L'export contient Markdown, JSON, SVG et un manifeste SHA-256. La commande
`verify` reconstruit ces fichiers et vérifie l'approbation et les sources.
Une version refusée ne peut pas être exportée. Pour de nouvelles sources ou
un nouveau protocole, créer une nouvelle revue afin de conserver l'historique.
