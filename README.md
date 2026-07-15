# 🩺 Breast Cancer Classification using Machine Learning

## 📖 Présentation

Ce projet a pour objectif de développer un modèle de **classification supervisée** capable de prédire si une tumeur mammaire est **bénigne** ou **maligne** à partir des caractéristiques morphologiques des cellules issues du jeu de données **Breast Cancer Wisconsin**.

Au-delà de la performance prédictive, ce travail met l'accent sur trois aspects essentiels de tout projet de Data Science : **la sélection des variables**, **l'optimisation des hyperparamètres** et **l'interprétabilité du modèle**. L'objectif est de construire un modèle à la fois performant, robuste et facilement explicable.

---

# 🎯 Objectifs

* Construire un modèle de classification fiable.
* Identifier les variables les plus discriminantes.
* Comparer plusieurs stratégies d'optimisation des hyperparamètres.
* Évaluer la robustesse du modèle par validation croisée.
* Interpréter les décisions du modèle grâce aux méthodes d'explicabilité.

---

# 📊 Jeu de données

Le projet utilise le jeu de données **Breast Cancer Wisconsin Diagnostic**, disponible dans **Scikit-learn**.

| Caractéristique       | Valeur                     |
| --------------------- | -------------------------- |
| Nombre d'observations | 569                        |
| Nombre de variables   | 30                         |
| Variable cible        | Diagnostic (Bénin / Malin) |
| Valeurs manquantes    | Aucune                     |

Chaque observation correspond aux mesures morphologiques extraites d'une image de biopsie mammaire.

---

# 🔄 Démarche du projet

Le projet suit les principales étapes d'un pipeline de Data Science :

1. Compréhension du problème métier
2. Analyse exploratoire des données (EDA)
3. Prétraitement des données
4. Analyse des corrélations
5. Sélection des variables
6. Construction des modèles
7. Optimisation des hyperparamètres
8. Validation croisée
9. Évaluation des performances
10. Interprétation des résultats

---

# 🔍 Analyse exploratoire

Une analyse exploratoire complète a été réalisée afin de :

* étudier la distribution des variables ;
* analyser les corrélations entre caractéristiques ;
* identifier les variables redondantes ;
* comprendre la séparation naturelle entre les deux classes.

Les principales visualisations comprennent :

* Distribution des classes
* Histogrammes des variables
* Matrice de corrélation
* Analyse en composantes principales (PCA)

---

# 🧹 Prétraitement des données

Les données ont été préparées avant l'entraînement des modèles :

* vérification des valeurs manquantes ;
* suppression des variables fortement corrélées ;
* normalisation lorsque nécessaire ;
* séparation des jeux d'entraînement et de test ;
* validation croisée stratifiée.

---

# 🎯 Sélection des variables

Plusieurs méthodes de sélection ont été comparées afin de réduire la dimension du problème tout en conservant les informations les plus pertinentes.

Les variables ont été sélectionnées à partir :

* de la **Mutual Information** ;
* de l'analyse des corrélations ;
* des importances calculées par les modèles.

Deux ensembles ont ensuite été comparés :

* un modèle utilisant **15 variables** ;
* un modèle utilisant **10 variables**.

---

# ⚙️ Optimisation des hyperparamètres

Trois approches ont été évaluées :

* **Randomized Search**
* **Optuna (Tree-structured Parzen Estimator)**
* **Bayesian Optimization (BayesSearchCV)**

L'optimisation bayésienne a obtenu les meilleurs résultats en offrant une recherche plus efficace et une meilleure stabilité des performances.

---

# 📈 Résultats

Les performances obtenues montrent qu'une réduction du nombre de variables permet de conserver un excellent niveau de précision tout en améliorant la simplicité du modèle.

| Modèle                                | Accuracy    | F1-score    | ROC-AUC    |
| ------------------------------------- | ----------- | ----------- | ---------- |
| Random Forest (15 variables)          | **95.23 %** | **95.73 %** | **0.9833** |
| Random Forest optimisé (10 variables) | **94.86 %** | **95.52 %** | **0.9881** |

Les différences observées étant faibles, le modèle utilisant **10 variables** a été retenu pour sa meilleure interprétabilité et sa complexité réduite.

---

# 🔬 Interprétabilité

L'interprétation du modèle constitue une étape essentielle de ce projet.

Les analyses réalisées comprennent :

* Feature Importance
* Importance moyenne des variables
* Analyse comparative entre les modèles
* Identification des variables les plus discriminantes

Les résultats mettent en évidence un noyau stable de variables contribuant fortement au diagnostic des tumeurs.

---

# 📌 Conclusions

Ce projet démontre qu'il est possible d'obtenir un modèle performant tout en limitant sa complexité. L'association d'une sélection rigoureuse des variables et d'une optimisation bayésienne des hyperparamètres permet de construire un modèle robuste, capable de maintenir un excellent niveau de performance avec seulement dix variables.

Cette approche améliore également l'interprétabilité du modèle, un aspect essentiel dans le domaine médical où la compréhension des décisions algorithmiques est aussi importante que leur précision.

---

# 🛠️ Technologies utilisées

* Python
* Pandas
* NumPy
* Scikit-learn
* Optuna
* Scikit-Optimize (BayesSearchCV)
* Matplotlib
* Seaborn


---

# 📂 Structure du projet ( N'es pas a jour)

```
📦 Breast-Cancer-Classification
│
├── data/
├── notebooks/
├── images/
├── models/
├── README.md
├── requirements.txt
└── Breast_Cancer.ipynb
```

---

# 🚀 Perspectives

Plusieurs pistes d'amélioration peuvent être envisagées :

* comparaison avec des modèles de boosting (CatBoost, LightGBM, XGBoost) ;
* validation sur des jeux de données cliniques externes ;
* calibration des probabilités de prédiction ;
* déploiement du modèle sous forme d'API ou d'application web interactive ;
* intégration d'une interface de visualisation destinée aux professionnels de santé.

---

# 👤 Auteur

**Ly**
Étudiant en Data Science & Data Analysis

Passionné par l'analyse de données, le machine learning et le développement de solutions basées sur l'intelligence artificielle, je développe des projets personnels afin de renforcer mes compétences et de construire un portfolio reflétant une démarche rigoureuse de bout en bout, de l'exploration des données jusqu'à l'interprétation des résultats.
