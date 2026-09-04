# Assistant RAG assurance — aide à la décision sinistres

Application Streamlit pour un gestionnaire de sinistres. On pose une question en
langage naturel ; l'assistant retrouve les clauses de contrat concernées, rédige
une réponse structurée avec un LLM (Groq, `gpt-oss-120b`), vérifie que chaque
citation figure bien dans les clauses sources, et renvoie vers une validation
humaine quand les éléments sont insuffisants.

Le corpus est entièrement synthétique : 15 contrats, 112 clauses, 12 branches,
générés par IA. Aucune donnée client réelle.

<p align="center">
  <img width="964" height="743" alt="image" src="https://github.com/user-attachments/assets/c676187e-34a4-4342-98ab-a28556973654" />

</p>

<p align="center">
  <img width="964" height="743" alt="image" src="https://github.com/user-attachments/assets/d1d34a77-df6e-49fb-9151-064ef3a84fa5" />

</p>

## Pourquoi ce projet

Sur un assistant contractuel, une réponse fausse mais plausible est plus
dangereuse qu'une absence de réponse. Deux erreurs reviennent : la recherche
remonte la mauvaise clause, ou le modèle invente une clause qui n'existe pas.

Le choix fait ici est un contrôle déterministe après génération : toute
affirmation qui ne se retrouve pas littéralement dans les clauses transmises au
modèle est rejetée. Pas de framework RAG non plus : chaque étape est écrite à la
main, ce qui la rend testable.

## Fonctionnalités

- Assistant conversationnel : plusieurs discussions, questions de suivi.
- Recherche hybride multilingue : embeddings `paraphrase-multilingual-MiniLM-L12-v2`
  (fastembed) plus un score lexical, avec filtre par branche.
- Réponse structurée : statut, décision, conditions, informations manquantes.
- Vérification des citations tolérante à la typographie (apostrophes, tirets,
  accents) mais pas aux reformulations.
- Bascule automatique vers une validation humaine quand les preuves sont trop faibles.

Un indicateur de fiabilité résume trois contrôles automatiques : informations
vérifiées (50 %), réponse adaptée aux preuves (30 %), usage des sources (20 %).
Il mesure le comportement des garde-fous, pas une probabilité juridique de prise
en charge.

## Pipeline

1. Chargement des contrats JSON et découpage par article.
2. Recherche hybride puis filtre par branche.
3. Génération d'un JSON structuré par le LLM.
4. Validation du schéma avec Pydantic.
5. Vérification de chaque citation dans les clauses sources.
6. Calcul de la fiabilité et garde-fou métier (escalade si besoin).

## Évaluation

`evaluate.py` mesure le pipeline sur 35 questions annotées couvrant les 12
branches, dont quelques cas piégeux (suicide la première année, défaut
d'entretien) et des ambiguïtés entre contrats proches. Recherche et génération
sont notées séparément, un échec de recherche étant la première cause
d'hallucination. Le jeu d'évaluation reste petit ; il faudrait l'étoffer pour des
chiffres vraiment représentatifs.

| Recherche  |      | Génération             |       |
| ---------- | :--: | ---------------------- | :---: |
| Hit-rate@5 | 97 % | Exactitude du statut   | 86 %  |
| Recall@5   | 96 % | Taux de citations      | 100 % |
| MRR        | 0.84 | Fidélité des citations | 100 % |
|            |      | Taux d'escalade        | 11 %  |

```bash
python3 evaluate.py                 # recherche + génération (si GROQ_API_KEY est défini)
python3 evaluate.py --no-llm        # recherche seule
python3 evaluate.py --report eval_report.md
```

La question qui échoue (maladie du chien pendant le délai de carence) confond la
clause de carence avec celle des frais vétérinaires. C'est typiquement le genre
de régression qu'une évaluation sert à repérer avant la mise en production.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip -r requirements.txt
```

Créer `.streamlit/secrets.toml` :

```toml
GROQ_API_KEY = "gsk_..."
```

Puis, depuis la racine :

```bash
streamlit run app.py
```

Il faut Python 3.10+ (3.12 conseillé), une clé API Groq, et un accès internet au
premier lancement pour télécharger le modèle d'embeddings.

## Organisation du code

```
app.py                          Interface Streamlit
evaluate.py                     Évaluation hors ligne
data/synthetic_contracts.json   Corpus synthétique
data/eval_questions.json        Questions annotées
src/chunking.py                 Chargement et découpage des contrats
src/models.py                   Modèles Pydantic
src/retriever.py                Recherche hybride
src/rag.py                      Génération, contrôles, garde-fous
```

## Limites connues

- Pas de gestion des versions de contrat, des dates d'effet, des pièces de
  sinistre ni des habilitations.
- La matrice d'embeddings tient en mémoire : suffisant pour les 112 clauses
  actuelles, pas pour un gros corpus.
- Les conversations ne sont pas persistées.
- Toute décision communiquée à un client doit être validée par un gestionnaire.
