import pickle
import numpy as np
import pandas as pd

EXTRACTED_FEATURES_SAVE_ADDR = "../extracted_features.pickle"


# read extracted features from pickle file
def read_extracted_features(extracted_features_save_adr):
    with open(extracted_features_save_adr, 'rb') as input_file:
        return pickle.load(input_file)


features = read_extracted_features(EXTRACTED_FEATURES_SAVE_ADDR)
features = dict(features)
averaged_features = {}

# keep the median for each patient patches
for patient in features.keys():
    averaged_features.update({patient: np.median(list(features[patient].values()), axis=0)})

# Convert the dictionary to a csv file where each row is a patient and each column is a feature using pandas
df = pd.DataFrame.from_dict(averaged_features, orient='index')
df.to_csv('../CSV/features.csv', index=True, header=True)

# Merge the extracted features with the clinical data on the patient ID
clinical_data = pd.read_csv('../CSV/clinical_data_with_no_missing_values.csv')
clinical_data = clinical_data.set_index('patient_id')
features = pd.read_csv('../CSV/features.csv')
features = features.set_index('Unnamed: 0')
features = clinical_data.join(features)
features.to_csv('../CSV/features_with_clinical_data_1024_2.csv', index=True, header=True)
