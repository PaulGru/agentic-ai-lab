# Pilote de recherche sur la relance d'outil

## Question
Une relance automatique après une erreur transitoire de lecture améliore-t-elle l'exactitude des réponses d'un agent ?

## Contexte
Nous voulons distinguer un échec temporaire de l'outil d'une erreur de réponse du modèle et préserver l'abstention quand la donnée est indisponible.

## Périmètre
Pilote contrôlé sur un catalogue fictif, avec et sans relance sous fonctionnement normal et sous erreur transitoire.

## Contraintes
Une seule relance de lecture, un snapshot de modèle fixe et un budget approuvé séparément pour la calibration et la mesure.

## Critères de réussite
Chaque réponse doit pouvoir être notée depuis les reçus bruts et chaque résultat du manuscrit doit référencer les essais correspondants.

## Exécuteur
retry_catalog
