# Rapport d'évaluation RAG

- Questions annotées : **35**
- Profondeur de recherche : **top-5**

## Retrieval (sans appel modèle)

| Métrique  | Valeur | Lecture                                      |
| ---------- | ------ | -------------------------------------------- |
| Hit-rate@5 | 97%    | au moins une clause attendue est remontée   |
| Recall@5   | 96%    | proportion des clauses attendues remontées  |
| MRR        | 0.84   | rang moyen de la première clause pertinente |

Génération (avec contrôles déterministes)

| Métrique            | Valeur | Lecture                                          |
| -------------------- | ------ | ------------------------------------------------ |
| Exactitude du statut | 86%    | décision conforme à l'attendu                  |
| Taux de citations    | 100%   | réponses appuyées par ≥ 1 extrait vérifié   |
| Fidélité           | 100%   | part des affirmations retrouvées littéralement |
| Taux d'escalade      | 11%    | réponses renvoyées vers un gestionnaire        |
