import warnings

import numpy as np
import pandas as pd
import umap
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load the CSV file
df = pd.read_csv('features_with_clinical_data_KNN.csv', delimiter=',', header=0)
print(df.shape)

# Select only the column OS and Follow-up Status for X1
X1 = df.iloc[:, 18:21]
X1 = X1.drop('PFS', axis=1)
print(X1.head())

# Select all the columns except the os and follow-up status
X2 = df.drop([df.columns[0], df.columns[18], df.columns[19] , df.columns[20]], axis=1)
print(X2.head())

# Use UMAP to reduce the number of features
reducer = umap.UMAP(n_components=63)
X2 = reducer.fit_transform(X2)
X2 = pd.DataFrame(X2)

# Concatenate the two arrays
X = np.hstack((X1, X2))
print(X.shape)

# Convert X to a dataframe
X = pd.DataFrame(X, columns=list(X1.columns) + list(X2.columns))
print(X.head())

# select the OS column as the outcome variable
y = df.iloc[:, 18]

X = X.astype('float32')
y = y.astype('float32')

# Cox regression
cph = CoxPHFitter(penalizer=0, alpha=0.05, strata=None)  # l1_ratio=0.1
cph.fit(X, duration_col='OS', event_col='Follow-up Status')
cph.print_summary()
cph.plot()
plt.show()

##############################
# We want to see the Survival curve for 5 patients
tr_rows = X.iloc[0:10, :]
print(tr_rows)
cph.predict_survival_function(tr_rows).plot()
plt.show()