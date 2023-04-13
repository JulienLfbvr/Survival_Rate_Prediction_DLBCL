# Importation des bibliothèques nécessaires
import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.impute import KNNImputer

# Définition du chemin du fichier csv contenant les données cliniques
clinical_data_path = 'D:\\ISEN\\M1\\Projet M1\\DLBCL-Morph\\clinical_data_cleaned.csv'

# Chemin du dossier contenant les images PNG
image_folder = 'D:\\ISEN\\M1\\Projet M1\\DLBCL-Morph\\Patches\\HE'

# Chargement des données cliniques dans un DataFrame
df_outcome = pd.read_csv(clinical_data_path)

# enlever les patients id qui ne sont pas dans les patches
for patient_id in df_outcome['patient_id'].unique():

    # Vérifier si un dossier existe pour le patient_id dans le dossier contenant les images
    patient_folder = os.path.join(image_folder, str(patient_id))
    if not os.path.exists(patient_folder):
        # supprimer la ligne dans le csv
        df_outcome = df_outcome.drop(df_outcome[df_outcome['patient_id'] == patient_id].index)

# Replace missing values of each column with interpolated values
# df_outcome = df_outcome.interpolate(method='linear', axis=0).ffill().bfill()

# K-NN imputation

# create a KNN imputer object
imputer = KNNImputer(n_neighbors=5)

# fit the imputer to the data and transform the DataFrame
df_outcome = pd.DataFrame(imputer.fit_transform(df_outcome), columns=df_outcome.columns)

# Affichage des 5 premières lignes du DataFrame
print(df_outcome.head())

# Sauvegarde des données cliniques avec les tableaux d'images liés dans un fichier csv
df_outcome.to_csv('../CSV/clinical_data_KNN.csv', index=False)
