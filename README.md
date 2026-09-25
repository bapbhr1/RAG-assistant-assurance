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

## Lien testable (Streamlit Cloud) : https://rag-assistant-assurance-f8bpqw6tve9jssay7gcuwg.streamlit.app/

Sur un assistant contractuel, une réponse fausse mais plausible est plus
dangereuse qu'une absence de réponse. Deux erreurs reviennent : la recherche
remonte la mauvaise clause, ou le modèle invente une clause qui n'existe pas.

Le choix fait ici est un contrôle déterministe après génération : toute
affirmation qui ne se retrouve pas littéralement dans les clauses transmises au
modèle est rejetée, tout comme un montant absent de la clause citée. Pas de
framework RAG non plus : chaque étape créée manuellement, ce qui la rend
testable (voir `tests/`).

## Fonctionnalités

- Assistant conversationnel : plusieurs discussions, questions de suivi.
- Recherche hybride multilingue : embeddings `paraphrase-multilingual-MiniLM-L12-v2`
  (fastembed) plus un score lexical, avec filtre par branche.
- Réponse structurée : statut, décision, conditions, informations manquantes.
- Vérification des citations tolérante à la typographie (apostrophes, tirets,
  accents) mais pas aux reformulations.
- Contrôle des montants : chaque montant, taux ou délai d'une affirmation doit
  figurer dans la clause citée. Une vraie citation ne peut pas couvrir un faux chiffre.
- Bascule automatique vers une validation humaine quand les preuves sont trop faibles.
- Traçage optionnel avec Langfuse : recherche, appel LLM et contrôles de chaque question.

Un indicateur de fiabilité résume trois contrôles automatiques : informations
vérifiées (50 %), réponse adaptée aux preuves (30 %), usage des sources (20 %).
Le deuxième critère note le statut proposé par le modèle avant le garde-fou : un
modèle trop affirmatif perd ces points même si le garde-fou corrige sa décision.
L'indicateur mesure le comportement des garde-fous, pas une probabilité juridique
de prise en charge.

## Pipeline

1. Chargement des contrats JSON et découpage par article.
2. Recherche hybride puis filtre par branche.
3. Génération d'un JSON structuré par le LLM.
4. Validation du schéma avec Pydantic.
5. Vérification de chaque citation et de ses montants dans les clauses sources.
6. Calcul de la fiabilité et garde-fou métier (escalade si besoin).

## Évaluation

`evaluate.py` mesure le pipeline sur 46 questions annotées couvrant les 12
branches : prises en charge, refus, cas piégeux (suicide la première année,
défaut d'entretien), prises en charge partielles (plafond dépassé), dossiers
incomplets et questions hors corpus. Ces deux dernières familles attendent une
validation par un gestionnaire : elles vérifient que l'assistant sait
s'abstenir. Recherche et génération sont notées séparément, un échec de
recherche étant la première cause d'hallucination. Le jeu reste petit ; il
faudrait l'étoffer pour des chiffres vraiment représentatifs.

| Recherche  |      | Génération             |       |
| ---------- | :--: | ---------------------- | :---: |
| Hit-rate@5 | 98 % | Exactitude du statut   | 87 %  |
| Recall@5   | 96 % | Fausse prise en charge | 5 %   |
| MRR        | 0.82 | Abstention correcte    | 89 %  |
|            |      | Escalade inutile       | 11 %  |
|            |      | Fidélité des citations | 99 %  |

La fausse prise en charge est l'erreur qui coûte : accorder une garantie que le
contrat refuse ou qui demande une vérification. Le seul cas relevé (capital
obsèques la première année) est une réponse « partielle » là où l'attendu est un
refus : seules les cotisations sont remboursées. Les escalades inutiles vont dans
le sens prudent. Le détail, avec la matrice de confusion, est dans
[`eval_report.md`](eval_report.md).

```bash
python3 evaluate.py                       # recherche + génération (si une clé Groq est trouvée)
python3 evaluate.py --no-llm              # recherche seule
python3 evaluate.py --compare-retrieval   # compare les méthodes de fusion
python3 evaluate.py --report eval_report.md
```

La clé est lue dans `GROQ_API_KEY` ou, à défaut, dans `.streamlit/secrets.toml`.

**Choix de la recherche.** La fusion pondérée dense + lexical a été comparée à
BM25 et à une fusion par rangs (RRF), l'approche la plus répandue :

| Fusion                  | Hit-rate@5 | MRR  |
| ----------------------- | :--------: | :--: |
| Pondérée (retenue)      | 98 %       | 0.82 |
| Dense seule             | 95 %       | 0.70 |
| BM25 seul               | 98 %       | 0.78 |
| RRF dense + BM25        | 98 %       | 0.79 |

Un reranker multilingue (`jina-reranker-v2-base-multilingual`) a aussi été testé
sur le top-20 : même MRR, pour 1,1 Go de modèle et une demi-seconde de plus par
question. Il n'a pas été retenu.

**Seuil de correspondance.** Les questions hors corpus obtiennent un score de
meilleure clause entre 0.53 et 0.68, les questions couvertes entre 0.59 et 0.89.
Les deux plages se chevauchent : le seuil (0.58) n'écarte qu'une partie des
questions hors sujet, les autres sont arrêtées par le modèle et le contrôle des
citations. L'abstention mesurée plus haut porte sur l'ensemble de la chaîne.

La question dont la recherche échoue (maladie du chien pendant le délai de
carence) confond la clause de carence avec celle des frais vétérinaires ; elle
part désormais en validation plutôt qu'en mauvaise réponse.

## Tests

Les contrôles déterministes sont couverts par des tests unitaires : tolérance
typographique des citations, rejet des reformulations et des montants inventés,
escalade, calcul de la fiabilité, découpage et recherche. Le LLM et le modèle
d'embeddings sont simulés : les tests tournent en moins d'une seconde, sans
réseau ni clé API. GitHub Actions les relance à chaque push.

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pytest
```

## Traçage 

Avec des clés [Langfuse](https://langfuse.com) dans `.streamlit/secrets.toml` ou
dans l'environnement, chaque question produit une trace : clauses remontées et
scores, prompt et réponse du modèle, tokens consommés, résultat des contrôles et
statut avant/après le garde-fou.

```toml
LANGFUSE_PUBLIC_KEY = "pk-lf-..."
LANGFUSE_SECRET_KEY = "sk-lf-..."
LANGFUSE_HOST = "https://cloud.langfuse.com"
```

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

Il faut Python 3.11+ (3.12 conseillé), clé API Groq / Langfuse (optionnel), et accès internet au
premier lancement pour télécharger le modèle d'embeddings.


## Organisation du code

```
app.py                          Interface Streamlit
evaluate.py                     Évaluation hors ligne
eval_report.md                  Dernier rapport d'évaluation
data/synthetic_contracts.json   Corpus synthétique
data/eval_questions.json        Questions annotées
src/chunking.py                 Chargement et découpage des contrats
src/models.py                   Modèles Pydantic
src/retriever.py                Recherche hybride
src/rag.py                      Génération, contrôles, garde-fous
src/tracing.py                  Traçage Langfuse optionnel
tests/                          Tests unitaires
```

## Limites connues

- Pas de gestion des versions de contrat, des dates d'effet, des pièces de
  sinistre ni des habilitations.
- La matrice d'embeddings tient en mémoire : suffisant pour les 112 clauses
  actuelles, pas pour un gros corpus.
- Le seuil de correspondance est calé sur le jeu d'évaluation lui-même ; il
  faudrait un jeu séparé pour le valider.
- Les réponses du modèle varient d'un passage à l'autre malgré une température
  nulle : les chiffres de génération bougent de quelques points entre deux
  évaluations.
- Les conversations ne sont pas persistées.
- Toute décision communiquée à un client doit être validée par un gestionnaire.
