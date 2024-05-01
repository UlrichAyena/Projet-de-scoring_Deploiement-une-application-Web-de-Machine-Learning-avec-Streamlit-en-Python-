
import os
import pandas as pd 
import numpy as np 
import streamlit as st 
import seaborn as sns 
import seaborn
import matplotlib.pyplot as plt 
import plotly.express as px
import missingno as msno
import tempfile
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,roc_curve, auc
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')
import joblib

# code streamlit

df=pd.read_csv("Dataset.csv")

st.sidebar.title("Projet de scoring")


# Affichage dans la barre latérale avec l'étiquette "Auteur"
#st.sidebar.markdown("<p style='font-weight:bold; color:black;'>Auteur :</p> <p style='color:blue;'> AYENA Mahougnon</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='font-weight:bold; color:black;'>Auteur :</p> <p style='color:blue;'><a href='https://www.linkedin.com/in/mahougnon-ayena'\
                     target='_blank'>AYENA Mahougnon</a></p>", unsafe_allow_html=True)


pages = ["Présentation du projet", "Le jeu de données", "Visualisation des données","Préparation des données" ,"Modélisation","Conclusion"]
page = st.sidebar.radio("Aller vers la page :", pages)

if page == pages[0] : 
    st.write("# Projet de scoring")
    st.image("Inclusion.jpeg")
    st.write("### Présentation du projet")
    
    st.write("Qui est le plus susceptible d'avoir un compte bancaire ? L'inclusion financière reste l'un des principaux obstacles au développement économique et humain en Afrique.\
              Par exemple, au Kenya, au Rwanda, en Tanzanie et en Ouganda, seuls 9,1 millions d'adultes \(soit 13,9 % de la population adulte) ont accès à un compte bancaire commercial ou l'utilisent. ")
    st.write(" Malgré la prolifération de l'argent mobile en Afrique, et la croissance des technologies innovantes. \
            Les banques continuent de jouer un rôle essentiel pour faciliter l'accès aux services financiers.")
    st.write(" L'accès à la banque permet aux ménages d'épargner et de faciliter les paiements tout en aidant\
            les entreprises à améliorer leur solvabilité et leur accès à d'autres services financiers.\
            Par conséquent, l'accès aux comptes bancaires est un facteur essentiel de la croissance économique à long terme.")
    st.write("#### Objectif du projet")
    st.write("Il s’agit de développer une solution de scoring permettant d’apporter une aide à la décision pour les chargés de relation client.\
              En effet, basée sur des algorithmes de Machine Learning, la solution calcule la probabilité qu’un client ait le droit d’ouvrir un compte bancaire ou non. ")
    st.write("#### Sommaire")
    st.write("Voici les différentes étapes suivies pour la réalisation de ce projet")
    st.write("###### 1-Lecture de la base de donnée")
    st.write("###### 2-Exploration des données")
    st.write("###### 3-Prétraitement des données")
    st.write("###### 4-Modélisation et Prediction")
    st.write("###### 5-Choix du bon modèle")
elif page == pages[1]:
    st.write("### Le jeu de données")
    st.write(" Les données que nous utilisons sont issues d'une enquête menée auprès de la population de quatre pays :\
              le Kenya, le Rwanda, la Tanzanie et l'Ouganda, sur une période de trois ans, de 2016 à 2018.\
              Au total, 23 524 personnes ont été interrogées dans le cadre de cette étude.")
    st.write("Voici les différentes variables de notre base de données")

# Création du DataFrame des variables
    data = {
        'Variable': ['country', 'year', 'uniqueid', 'bank_account', 'location_type', 
                 'cellphone_access', 'household_size', 'age_of_respondent', 
                 'gender_of_respondent', 'relationship_with_head', 'marital_status', 
                 'education_level', 'job_type'],
        'Description': ['Pays associé à chaque observation dans l\'ensemble de données.',
                    'Année à laquelle chaque observation a été enregistrée.',
                    'Identifiant unique pour chaque observation.',
                    'Indique si un individu possède un compte bancaire.',
                    'Type de localisation où réside chaque individu (urbain ou rural).',
                    'Indique si un individu a accès à un téléphone portable.',
                    'Taille du ménage auquel chaque individu appartient.',
                    'Âge de chaque individu enregistré.',
                    'Genre de chaque individu (homme ou femme).',
                    'Relation de chaque individu avec le chef du ménage.',
                    'État civil de chaque individu (marié, célibataire, etc.).',
                    'Niveau d\'éducation atteint par chaque individu.',
                    'Type d\'emploi ou d\'activité professionnelle de chaque individu.'],
        'Type': ['Qualitatif', 'Numérique', 'Qualitatif', 'Qualitatif binaire', 'Qualitatif binaire', 
                 'Qualitatif binaire', 'Numérique', 'Numérique', 
                 'Qualitatif binaire', 'Qualitatif', 'Qualitatif', 
                 'Qualitatif', 'Qualitatif']
    }
    data0 = pd.DataFrame(data)

# Affichage du DataFrame dans Streamlit
    st.dataframe( data0)


elif page == pages[2]:
     st.write("### Visualisation des données")
    
     st.dataframe(df.head())
    
     st.write("Dimensions du dataframe :")
    
     st.write(df.shape)
 ##---------------------------------------------------------------------------#   
     st.write("###### Le nombre de variables quantitatives et de variables qualitatives de notre base de données")
    
    # Calcul du nombre de variables catégorielles et numériques
     cat_count = len(df.select_dtypes(include=['object']).columns)
     num_count = len(df.select_dtypes(include=['int64','float64']).columns)
    
    # Affichage du nombre de variables catégorielles et numériques
     st.write(f"Nombre total de variables : {cat_count + num_count}")
     st.write(f"Variables catégorielles : {cat_count}")
     st.write(f"Variables numériques : {num_count}")
    
    # Affichage des noms de toutes les variables catégorielles
     st.write("###### Variables catégorielles :")
     for col in df.select_dtypes(include=['object']).columns.tolist():
         st.write(f"- {col}")
    
    # Affichage des noms de toutes les variables numériques
     st.write("###### Variables numériques :")
     for col in df.select_dtypes(include=['int64','float64']).columns.tolist():
        st.write(f"- {col}")
##---------------------------------------------------------------------------#
     if st.checkbox("Afficher les valeurs manquantes") : 
        st.dataframe(df.isna().sum())
     #if st.checkbox("Visualisation des données manquantes"):
        fig = msno.matrix(df)
        st.pyplot(fig.figure)
        
     if st.checkbox("Afficher les doublons") : 
        st.write(df.duplicated().sum())

##---------------------------------------------------------------------------#
   # Chargement des données
     dataset1 = df.drop("uniqueid", axis=1)
   # Sélecteur pour choisir la variable qualitative
     selected_variable = st.selectbox('Choisir une variable qualitative :', dataset1.select_dtypes('object').columns)

   # Vérifier si une variable est sélectionnée
     if selected_variable:
    # Calculer les effectifs de chaque modalité
        value_counts = dataset1[selected_variable].value_counts()
    
    # Calcul du nombre total d'observations dans la colonne
        total_observations = value_counts.sum()
    
    # Calcul des pourcentages de chaque modalité
        percentages = (value_counts / total_observations) * 100
    
    # Création d'un DataFrame pour stocker les effectifs et les pourcentages
        df_stats = pd.DataFrame({'Effectif': value_counts, 'Pourcentage (%)': percentages})
    
    # Affichage du nom de la colonne et ses valeurs uniques dans Streamlit
        st.write(f'### {selected_variable}')
        st.dataframe(df_stats)

##---------------------------------------------------------------------------#

# Liste des variables qualitatives sauf 'uniqueid'
     qualitative_variables = [col for col in df.select_dtypes(include=['object']) if col != 'uniqueid']

# Sélecteur pour choisir la variable qualitative
     selected_variable = st.selectbox("Sélectionnez une variable qualitative :", qualitative_variables)

# Vérifier si une variable a été sélectionnée
     if selected_variable:
    # Calcul des effectifs de chaque modalité
        value_counts = df[selected_variable].value_counts()

    # Création du diagramme en bâtons
        plt.figure(figsize=(8, 6))
        sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')

    # Ajout des titres et des étiquettes
        plt.title(f'Diagramme en bâtons pour la variable {selected_variable}')
        plt.xlabel(selected_variable)
        plt.ylabel('Effectif')

        # Faire pivoter les étiquettes de l'axe x pour une meilleure lisibilité
        plt.xticks(rotation=45, ha='right')

        # Affichage du diagramme
        st.pyplot(plt)

##---------------------------------------------------------------------------#
# Liste des choix de variables
     choices = [
         {'variable_qualitative': 'bank_account', 'variable_temporelle': 'year', 'title': 'Création de comptes bancaires en fonction des années'},
         {'variable_qualitative': 'bank_account', 'variable_temporelle': 'gender_of_respondent', 'title': 'Création de comptes bancaires en fonction du sex des individus'},
         {'variable_qualitative': 'bank_account', 'variable_temporelle': 'education_level', 'title': 'Création de comptes bancaires en fonction du niveau d\'éducation  des individus'},
        {'variable_qualitative': 'bank_account', 'variable_temporelle': 'country', 'title': 'Création de comptes bancaires en fonction des pays de l\'étude'}
       ]

   # Sélection de la variable à explorer
     selected_choice = st.selectbox("Sélectionnez une variable à explorer :", [choice['title'] for choice in choices])

  # Trouver le choix sélectionné
     selected_choice_data = next(choice for choice in choices if choice['title'] == selected_choice)
  
  # Création du graphique pour le choix sélectionné
     #plt.figure(figsize=(10, 6))
     fig, ax = plt.subplots(figsize=(15, 8))
     sns.countplot(x=selected_choice_data['variable_temporelle'], hue=selected_choice_data['variable_qualitative'], data=df, palette='Set2', ax=ax)

   # Ajout des titres et des étiquettes
     #plt.title(selected_choice_data['title'])
     #plt.xlabel(selected_choice_data['variable_temporelle'].capitalize())
     #plt.ylabel('Effectif')
     ax.set_title(selected_choice_data['title'])
     ax.set_xlabel(selected_choice_data['variable_temporelle'].capitalize())
     ax.set_ylabel('Effectif')

  # Affichage du graphique dans Streamlit
     st.pyplot(fig)
elif page == pages[3]:
     st.write("### Préparation des données")
     st.write("Dans cette section, nous explorerons les différents traitements que nous avons appliqués au jeu \
              de données pour le nettoyer et le préparer pour les étapes suivantes.")
     st.write("##### Démarche utilisée")
     st.write("###### Méthode LabelEncoder")
     st.write("Dans un premier temps, nous avons appliqué une étape de labélisation (LabelEncoder) aux variables qualitatives présentant uniquement deux modalités.\
               Cette étape consiste à attribuer à chaque modalité un nombre entier unique, permettant ainsi de transformer les données catégorielles en données\
               numériques. Pour ce faire, nous avons utilisé la classe LabelEncoder de la bibliothèque scikit-learn. Voici une illustration de cette démarche")
     st.image("labelencoding.png")
     st.write("###### Voici les variables auxquelles cette méthode a été appliquée:")
     st.write(" - **bank_account**: Indique si un individu possède un compte bancaire")
     st.write(" - **location_type**: Type de localisation où réside chaque individu (urbain ou rural)")
     st.write(" - **cellphone_access**: Indique si un individu a accès à un téléphone portable")
     st.write(" - **gender_of_respondent**: Genre de chaque individu (homme ou femme)")
     st.write("###### Méthode pandas get dummies")
     st.write("Ensuite, nous avons utilisé la méthode pd.get_dummies() pour créer des variables indicatrices (dummy variables) à partir des autres variables \
              catégorielles ayant plus de deux modalités dans le DataFrame. Cette méthode consiste à créer de nouvelles colonnes binaires pour chaque modalité de la\
               variable catégorielle, où chaque colonne représente une modalité avec la valeur 1 si l'observation correspondante appartient à cette modalité, et 0 \
              sinon. Cette approche permet de traiter correctement les variables catégorielles avec plusieurs modalités dans les modèles d'apprentissage automatique, \
              sans introduire de biais en attribuant un ordre arbitraire aux catégories. Voici un exemple de cette approche :")
     st.image("Getdummies.png")
     st.write("###### Voici les variables auxquelles cette méthode a été appliquée:")
     st.write(" - **country**: Pays associé à chaque observation dans l'ensemble de données")
     st.write(" - **relationship_with_head**: Relation de chaque individu avec le chef du ménage")
     st.write(" - **marital_status**: État civil de chaque individu (marié, célibataire, etc.)")
     st.write(" - **education_level**: Niveau d'éducation atteint par chaque individu")
     st.write(" - **job_type**: Type d'emploi ou d'activité professionnelle de chaque individu")
     st.write(" Suite à l'application de ces deux méthodes sur les variables qualitatives, le nombre total de variables dans notre base de données a augmenté de 13 à 38.")
elif page == pages[4]:
     st.write("### Modélisation")
     st.write("Dans cette phase de modélisation, nous avons divisé nos données en **ensembles d'apprentissage** et **de test** pour évaluer les performances des modèles. Ensuite,\
               nous avons normalisé les variables continues pour garantir une mise à l'échelle uniforme des données. Trois modèles différents, à savoir la **régression \
              logistique**, la **forêt aléatoire** et **XGBoost**, ont été entraînés sur les données d'apprentissage. Les performances de chaque modèle ont été évaluées en calculant \
              **l'Aire sous la courbe ROC (AUC)** sur l'ensemble de test. Enfin, nous avons utilisé la **validation croisée** pour ajuster les hyperparamètres des modèles et avons \
              tracé les courbes ROC pour visualiser leurs performances respectives.")
     
     df_prep = pd.read_csv("nvelle_base.csv")
     VARIABLE_CONTINUES=["household_size","age_of_respondent","year"]
     TARGET='bank_account'
    
     y = df_prep[TARGET]
     X= df_prep.drop(TARGET, axis=1)
     

     ## Division en train et en test
     X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=123)
    
     ##Normalisation
     scaler = StandardScaler()
     X_train[VARIABLE_CONTINUES] = scaler.fit_transform(X_train[VARIABLE_CONTINUES])
     X_test[VARIABLE_CONTINUES] = scaler.transform(X_test[VARIABLE_CONTINUES])
     

     ### Lecture des fichiers des modèles                         
     #reg_log = joblib.load("models/Logistic Regression_best_model.joblib")
     #rf = joblib.load("models/Random Forest_best_model.joblib")
     #xgboost = joblib.load("models/XGBoost_best_model.joblib")
     # Récupérer le chemin absolu du répertoire courant où se trouve ce script
     current_directory = os.path.dirname(__file__)

    # Concaténer le chemin absolu avec le chemin relatif vers le dossier des modèles
     models_directory = os.path.join(current_directory, "models")

    # Définir une liste des noms de fichiers de modèle
     model_files = ["Logistic Regression_best_model.joblib", 
               "Random Forest_best_model.joblib", 
               "XGBoost_best_model.joblib"]

     # Charger tous les modèles et les stocker dans un dictionnaire
     models = {}
     for model_file in model_files:
         model_path = os.path.join(models_directory, model_file)
         model_name = model_file.split('.')[0].replace(' ', '_')  # Nom du modèle sans l'extension
         models[model_name] = joblib.load(model_path)

     # Accéder aux modèles chargés
     reg_log = models["Logistic_Regression_best_model"]
     rf = models["Random_Forest_best_model"]
     xgboost = models["XGBoost_best_model"]



     # Fonction pour calculer et afficher les performances des modèles
     def display_model_performance(model, model_name):
    # Prédire les probabilités de classe 1 sur l'ensemble de données de test
         y_probs = model.predict_proba(X_test)[:, 1]
    # Prédire les classes
         y_pred = model.predict(X_test)
    # Calculer le taux de faux positifs (FPR), le taux de vrais positifs (TPR) et l'AUC
         fpr, tpr, _ = roc_curve(y_test, y_probs)
         auc_score = auc(fpr, tpr)
    # Calculer la matrice de confusion
         cm = confusion_matrix(y_test, y_pred)
    # Afficher les performances du modèle
         st.write(f"##### Performance du modèle {model_name}")
         st.write(f"AUC Score: {auc_score:.2f}")
    # Afficher la matrice de confusion
         st.write("Tracer la courbe ROC:")
         #st.write(cm)
    # Tracer la courbe ROC
         plt.figure(figsize=(8, 6))
         plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")
         plt.plot([0, 1], [0, 1], linestyle='--', color='orange')
         plt.xlabel('Taux de faux positifs')
         plt.ylabel('Taux de vrais positifs')
         plt.title(f'Courbe ROC pour le modèle {model_name}')
         plt.legend(loc='lower right')
         plt.grid(True)
         st.pyplot(plt)

         # Ajout d'un espace
         st.write("")
         # Ajout d'un espace
         st.write("")
         # Ajout d'un espace
         st.write("")
         # Ajout d'un espace
         st.write("")



         #------------------------
         st.write("Matrice de confusion :")
    # Afficher la matrice de confusion sous forme de heatmap
         plt.figure(figsize=(26, 6))
         plt.subplot(1, 2, 2)
         sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
         plt.xlabel('Valeurs prédites')
         plt.ylabel('Vraies valeurs')
         #plt.title('Matrice de confusion')
         st.pyplot(plt.gcf())


    # Interface utilisateur Streamlit
     st.write("##### Évaluation des modèles")
     selected_model = st.selectbox("**Sélectionnez un modèle:**", ["Régression Logistique", "Forêt Aléatoire", "XGBoost"])

     if selected_model == "Régression Logistique":
              display_model_performance(reg_log, "Régression Logistique")
     elif selected_model == "Forêt Aléatoire":
              display_model_performance(rf, "Forêt Aléatoire")
     else:
             display_model_performance(xgboost, "XGBoost")
     models = {
     "Régression Logistique": reg_log,
     "Forêt Aléatoire": rf,
     "XGBoost": xgboost
     }
     # Ajout d'un espace
     st.write("")
     # Ajout d'un espace
     st.write("")
     # Ajout d'un espace
     st.write("")
     # Ajout d'un espace
     st.write("")



##--------------------------------------------------------------------------------------------------------------------------
     st.write("##### Tracé de la courbe de ROC pour l'ensemble des modèles")
     # Création d'une liste pour stocker les modèles et leur nom
     all_models = [("Régression Logistique", reg_log), ("Forêt Aléatoire", rf), ("XGBoost", xgboost)]

     # Création d'une figure
     plt.figure(figsize=(10, 8))

     # Boucle sur chaque modèle pour calculer et tracer la courbe ROC
     for model_name, model in all_models:
        # Prédiction des probabilités de classe 1 sur l'ensemble de données de test
        y_probs = model.predict_proba(X_test)[:, 1]
        # Calcul du taux de faux positifs (FPR), du taux de vrais positifs (TPR) et de l'AUC
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc_score = auc(fpr, tpr)
        # Tracé de la courbe ROC
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_score:.2f})")

      # Ajout de la ligne en pointillés diagonale (aucune compétence)
     plt.plot([0, 1], [0, 1], linestyle='--', color='orange')

    # Ajout des labels et du titre
     plt.xlabel('Taux de faux positifs')
     plt.ylabel('Taux de vrais positifs')
     plt.title('Courbe ROC pour tous les modèles')
     plt.legend(loc='lower right')
     plt.grid(True)

    # Affichage du graphique
     st.pyplot(plt)
    # Ajout d'un espace
     st.write("")
##-----------------------------------------------------------------------------------------------------------------
    # Création d'une liste pour stocker les modèles et leur nom
     all_models = [("Régression Logistique", reg_log), ("Forêt Aléatoire", rf), ("XGBoost", xgboost)]

   # Création d'un dictionnaire pour stocker les AUC de chaque modèle
     auc_scores = {}

    # Boucle sur chaque modèle pour calculer les courbes ROC et les AUC
     for model_name, model in all_models:
    # Prédiction des probabilités de classe 1 sur l'ensemble de données de test
      y_probs = model.predict_proba(X_test)[:, 1]
    # Calcul du taux de faux positifs (FPR), du taux de vrais positifs (TPR)
      fpr, tpr, _ = roc_curve(y_test, y_probs)
      auc_score = auc(fpr, tpr)
    # Stockage de l'AUC dans le dictionnaire
      auc_scores[model_name] = auc_score

    # Recherche du meilleur modèle avec le score AUC le plus élevé
     best_model_name = max(auc_scores, key=auc_scores.get)
     best_model_auc = auc_scores[best_model_name]
     # Ajout d'un espace
     st.write("")
    # Affichage du meilleur modèle
     st.write(f"#### Meilleur modèle: {best_model_name}")
     st.write(f"**Score AUC**: {best_model_auc:.2f}")
elif page == pages[5]:
     st.write("### Conclusion")
     st.write("Dans l'ensemble, notre travail a été axé sur l'exploration, le nettoyage, la modélisation et l'évaluation des performances de différents modèles pour prédire la possession d'un compte bancaire.\
               Nous avons commencé par explorer et nettoyer les données, en traitant les valeurs manquantes et en transformant les variables catégorielles. Ensuite, nous avons construit plusieurs modèles de machine\
               learning, notamment la régression logistique, la forêt aléatoire et XGBoost, pour prédire la variable cible. Nous avons évalué les performances de ces modèles en utilisant des métriques telles que l'AUC ROC.\
               Nos résultats ont montré que le modèle XGBoost a généralement produit les meilleures performances parmi les trois modèles testés, avec un score AUC plus élevé. Pour les perspectives futures, nous pourrions \
              explorer d'autres algorithmes de machine learning, effectuer une optimisation plus avancée des hyperparamètres et peut-être recueillir davantage de données pour améliorer encore les performances du modèle.")
