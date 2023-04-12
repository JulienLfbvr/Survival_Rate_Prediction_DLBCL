# Importation des bibliothèques nécessaires
import os
import pandas as pd
from PIL import Image
import numpy as np

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

# fill NaN entries with the mean value of the feature
df_outcome = df_outcome.fillna(df_outcome.mean())

# arrondir les moyennes à trois chiffres après la virgule
df_outcome = df_outcome.round(3)

# Affichage des 5 premières lignes du DataFrame
print(df_outcome.head())

# Sauvegarde des données cliniques avec les tableaux d'images liés dans un fichier csv
df_outcome.to_csv('clinical_data_with_no_missing_values.csv', index=False)
