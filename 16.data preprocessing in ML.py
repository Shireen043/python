import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv(r"C:\Users\acer\Downloads\Data (1).csv")

X = dataset.iloc[:, :-1].values

Y = dataset.iloc[: , 3].values

# sklearn fill missing numerical value

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")

#imputer = SimpleImputer()

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

# impute categorical value for independent

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0])

X[:,0] = labelencoder_X.fit_transform(X[:,0])

# IMPUTE CATEGORICAL VALUE FOR INDEPENDENT

labelencoder_Y = LabelEncoder()

Y = labelencoder_Y.fit_transform(Y)

# SPLIT THE DATA

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2, random_state=0)
