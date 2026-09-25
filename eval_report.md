# Rapport d'évaluation RAG

- Questions annotées : **46** (27 prises en charge, 2 partielles, 8 refus, 9 à vérifier dont 5 hors corpus)
- Profondeur de recherche : **top-5**

## Retrieval (sans appel modèle)

Mesuré sur les 41 questions qui ont au moins une clause attendue.

| Métrique | Valeur | Lecture |
| --- | --- | --- |
| Hit-rate@5 | 98% | au moins une clause attendue est remontée |
| Recall@5 | 96% | proportion des clauses attendues remontées |
| MRR | 0.82 | rang moyen de la première clause pertinente |

Questions sans clause pertinente dans le top-5 : animaux-carence-maladie.

### Score de la meilleure clause

| Questions | min / médiane / max |
| --- | --- |
| Avec clause attendue | 0.59 / 0.76 / 0.89 |
| Hors corpus | 0.53 / 0.65 / 0.68 |

En dessous de 0.58, la réponse part en validation humaine.

### Comparaison des méthodes de fusion

| Fusion | Hit-rate@5 | Recall@5 | MRR |
| --- | --- | --- | --- |
| Pondérée dense + lexical (défaut) | 98% | 96% | 0.82 |
| Dense seule | 95% | 93% | 0.70 |
| BM25 seul | 98% | 98% | 0.78 |
| RRF dense + BM25 | 98% | 95% | 0.79 |

## Génération (avec contrôles déterministes)

| Métrique | Valeur | Lecture |
| --- | --- | --- |
| Exactitude du statut | 87% | décision conforme à l'attendu |
| Fausse prise en charge | 5% | garantie accordée à tort, parmi les questions qui ne sont pas des prises en charge complètes |
| Abstention correcte | 89% | questions à vérifier bien renvoyées vers un gestionnaire |
| Escalade inutile | 11% | questions tranchables renvoyées vers un gestionnaire |
| Taux de citations | 97% | questions tranchables appuyées par ≥ 1 extrait vérifié |
| Fidélité | 99% | affirmations retrouvées littéralement, montants compris |
| Taux d'escalade | 26% | réponses renvoyées vers un gestionnaire |

### Matrice de confusion

| Attendu \ Obtenu | Pris en charge | Partiel | Non pris en charge | À vérifier |
| --- | --- | --- | --- | --- |
| Pris en charge | 24 | 0 | 0 | 3 |
| Partiel | 0 | 2 | 0 | 0 |
| Non pris en charge | 0 | 1 | 6 | 1 |
| À vérifier | 0 | 0 | 1 | 8 |

Écarts avec l'attendu :

- hab-chien-tiers : attendu Pris en charge, obtenu À vérifier
- pro-perte-exploitation : attendu Pris en charge, obtenu À vérifier
- animaux-carence-maladie : attendu Non pris en charge, obtenu À vérifier
- animaux-forfait-prevention : attendu Pris en charge, obtenu À vérifier
- obseques-carence-premiere-annee : attendu Non pris en charge, obtenu Partiel
- hors-corpus-smartphone : attendu À vérifier, obtenu Non pris en charge
