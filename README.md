# Prédiction de l'Abandon Client (Churn Prediction)

Ce projet vise à prédire la probabilité qu'un client abandonne un service (churn) en utilisant des techniques de machine learning.

## Objectifs

- Analyser les données clients pour identifier les facteurs d'abandon
- Construire un modèle de classification binaire performant
- Identifier les caractéristiques les plus importantes dans la décision d'abandon
- Visualiser les résultats et les profils à risque

## Structure du Projet

```
.
├── data/               # Dossier contenant les datasets
├── churn_prediction.py # Script principal
├── requirements.txt    # Dépendances Python
└── README.md          # Documentation
```

## Installation

1. Cloner le repository :
```bash
git clone [URL_DU_REPO]
cd churn-prediction
```

2. Créer un environnement virtuel et installer les dépendances :
```bash
python -m venv venv
source venv/bin/activate  # Sur Unix/macOS
# ou
.\venv\Scripts\activate  # Sur Windows
pip install -r requirements.txt
```

## Utilisation

1. Télécharger le dataset Telco Customer Churn depuis Kaggle et le placer dans le dossier `data/`
2. Exécuter le script principal :
```bash
python churn_prediction.py
```

## Fonctionnalités

- Prétraitement des données
- Feature engineering
- Gestion des données déséquilibrées (SMOTE)
- Modèles de classification (Random Forest, XGBoost)
- Évaluation des performances (AUC, précision, rappel)
- Visualisation des résultats

## Auteur

[Votre Nom]

## Licence

MIT 