# Exemples V8.6 — entièrement fictifs

Ces documents démontrent les composants du scientifique autonome. Ils ne sont
pas issus d'une campagne de recherche, de Jean Zay ou d'articles vérifiés.

- `v85_paired_fixture.json` contient les scores synthétiques V8.5.
- `v86_design_context.json` rassemble une mission, une littérature et un design
  de test. Les notices, DOI, extraits et approbations sont **fictifs** et repris
  des fixtures unitaires. Aucun accès OpenAlex n'a été effectué pour les créer.
- `v86_review_input.json` configure une revue sans LLM et deux cycles maximum.
- `v86_revision_draft.json` propose de préciser une limite méthodologique. Son
  lien à la critique `assumed_independence` justifie le changement de texte.

Le design et les scores ne sont **pas reliés comme protocole et résultats d'une
même expérience**. La révision indique explicitement cette limite. Les hashes
cohérents des exemples ne prouvent aucune approbation humaine réelle.

Depuis la racine du dépôt, dans l'environnement Python du projet :

```powershell
python -m scripts.tutorials.scientific_critic preview --input examples/v86_review_input.json
python -m scripts.tutorials.scientific_critic offline --input examples/v86_review_input.json --draft examples/v86_revision_draft.json
```

La seconde commande demande `APPROVE` avant d'adopter la nouvelle version ;
toute autre réponse la refuse. Elle ne lance aucun job. Sans `--draft`, elle
produit seulement la revue. Pour une nouvelle revue, copier le fichier de
configuration et changer `review_id` ; conserver les historiques existants.
