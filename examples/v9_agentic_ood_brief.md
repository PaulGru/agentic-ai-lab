# Pilote descriptif de robustesse agentique

## Question
Sur des lectures de catalogue appariées, comment la réussite varie-t-elle entre les checkpoints Dense et MoE disponibles, les politiques de relance et les perturbations contrôlées ?

## Contexte
La tâche exige de consulter un outil local de catalogue. Le pilote compare les comportements de Qwen3-1.7B, Qwen3-4B-Instruct-2507 et OLMoE-1B-7B-Instruct avec les mêmes entrées, le même outil et un budget commun.

## Périmètre
Comparer une lecture unique et une relance après panne transitoire. Évaluer le contrôle, la reformulation, la panne transitoire et un texte de distraction. Les trois tâches et les deux ordres d'essais viennent du corpus versionné.

## Contraintes
Fixer les révisions et les paramètres d'inférence. Compter comme réussite une lecture valide suivie de la réponse entière exacte au format prescrit. Conserver les abstentions et les réponses hors format. Présenter les contrastes entre checkpoints comme descriptifs, avec les différences d'entraînement et de taille déclarées.

## Critères de réussite
Produire les trois reçus GPU, calculer les écarts appariés à partir des traces et vérifier la sensibilité des résultats au retrait d'une tâche. Relier chaque résultat du brouillon ACL aux données d'origine.
