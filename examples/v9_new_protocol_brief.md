# Comparer deux heuristiques de classement

## Question
Le score pondéré classe-t-il mieux les documents pertinents que le simple recouvrement de termes ?

## Contexte
Chaque tâche présente deux documents, A et B. Des caractéristiques entières fournissent le nombre de termes communs et un score pondéré pour chaque document. La référence indique le document jugé le plus pertinent.

## Périmètre
Comparer les modalités « overlap » et « weighted » du facteur « method » sur trois paires fictives appariées. En cas d'égalité, choisir A. Les indices 1 et 2 désignent A et B.

## Contraintes
Utiliser les mêmes paires dans les deux conditions. Fixer les graines 13 et 42. Le pilote fonctionne hors réseau pour la mesure et utilise le corpus local d'évaluation fourni par l'opérateur.

## Critères de réussite
Calculer la différence d'exactitude en points de pourcentage à partir des reçus et relier les résultats du brouillon aux essais. Présenter le constat comme descriptif.
