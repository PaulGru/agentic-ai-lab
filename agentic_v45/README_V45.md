# Agentic AI Lab — V4.5

## Objet

Consolidation du parcours **router → agent → handoff → agent délégué → retour → politique d'approbation → validation humaine → outil → réponse** dans un seul `AgentToolGraph`.

Nouvelles protections illustrées :
- les états des providers sont séparés par agent dans `provider_states` ;
- `handoff_stack` conserve l'identité de l'appel et empêche une boucle de délégation ;
- la politique est revérifiée **juste avant** l'exécution ;
- un outil non enregistré est toujours refusé ;
- `CreateNoteOnceTool` crée un fichier une seule fois avec un ID déterministe ;
- une répétition avec le même ID et contenu est sans effet supplémentaire ;
- un ID déjà présent avec un contenu différent provoque une erreur.

## Installation

Depuis la racine du dépôt, extraire l'archive puis :

```bash
cp agentic_v45/orchestration/agent_tool_graph.py orchestration/
cp agentic_v45/tools/create_note_once.py tools/
cp agentic_v45/tests/stateful_provider.py tests/
cp agentic_v45/tests/test_v45_resilience.py tests/
cp agentic_v45/demo_v45.py .
```

`approval_policy.py` et les providers restent inchangés.

Ajouter `v45_notes/` à `.gitignore`. Les `*.sqlite` sont déjà ignorés.

## Démonstration gratuite (FakeProvider)

```bash
python demo_v45.py pause v45_ok
python demo_v45.py approve v45_ok
cat v45_notes/v45_ok.txt
python demo_v45.py replay v45_ok

python demo_v45.py pause v45_no
python demo_v45.py reject v45_no
test ! -e v45_notes/v45_no.txt && echo 'refus respecté'

python demo_v45.py pause v45_edit
python demo_v45.py edit v45_edit
cat v45_notes/v45_edit.txt
```

Utiliser un **nouveau thread_id** à chaque `pause`. Le script sauvegarde les états dans `checkpoints_v45.sqlite` et reconstruit des objets providers vides lors de la reprise. Il n'appelle aucune API payante.

## Tests

```bash
python -m unittest discover -s tests -v
```

Les nouveaux tests comprennent la reprise après recréation des agents/providers, le non-routage à la reprise, les handoffs, le refus, le changement de politique, l'idempotence, les conflits, les corrections invalides et les appels inconnus.

## Limites à retenir

1. `CreateNoteOnceTool` démontre une forme d'**idempotence métier**, pas une garantie globale *exactly once*. En cas de crash pendant l'écriture, un fichier partiel peut exister : la reprise refuse de l'écraser et une réparation humaine peut être nécessaire.
2. `AgentRunner.run()` et l'ancien `GraphOrchestrator` peuvent toujours exécuter leurs outils sans cette supervision. Pour les outils sensibles, faire passer les actions exclusivement par `AgentToolGraph`.
3. Le programme appelant doit authentifier les décideurs, verrouiller les reprises concurrentes d'un même thread et protéger les fichiers SQLite.
4. Les identifiants de reprise OpenAI doivent rester utilisables côté API. Les appels API sortants eux-mêmes ne sont pas garantis exactement une fois.
5. Ce graphe change la topologie des nœuds de V4.4 : utiliser des **nouveaux threads/checkpoints**, ne pas relancer les anciens workflows interrompus avec cette version.
