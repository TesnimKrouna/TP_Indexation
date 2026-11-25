# TP4 — BM25 et évaluation

## Fichiers
- `documents.tsv` : les 25 documents fournis, plus D26-D30 pour les exercices 5 et 6.
- `tp4.py` : programme complet avec BM25 manuel, TF-IDF, recherche booléenne, métriques, expériences, grille et moteur interactif.
- `queries.tsv` : 8 requêtes de travail.
- `qrels.tsv` : jugements initiaux de pertinence.
- `requirements.txt` : bibliothèques utiles.

## Installation et lancement
```bash
python -m pip install -r requirements.txt
python tp4.py
```
`matplotlib` sert aux graphiques et `scipy` à Spearman. Le cœur BM25 et les métriques ne dépendent d'aucune bibliothèque externe.

## Important concernant Qrels
Le fichier TP3 fourni ne précisait pas les cinq requêtes de test antérieures. Les cinq premières requêtes de `queries.tsv` sont donc des requêtes de travail adaptées au corpus, suivies de trois nouvelles. Les grades sont des jugements initiaux établis à partir du contenu des documents, avant l'exécution des modèles. Vérifie-les et adapte-les si ton enseignante attend les requêtes exactes de TP3.

## Documents ajoutés
- D26, D27, D28 : `blockchain` apparaît respectivement 1, 3 et 10 fois, sur des documents de même longueur.
- D29, D30 : `blockchain` apparaît 3 fois, avec une longueur de 6 et 60 termes.
Le mini-corpus de l'exercice 4 reste indépendant.
