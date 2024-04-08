#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Load the dataset
data = pd.read_csv("data/ModeChoiceOptima.txt", sep='\t')  # Adjust the file path accordingly

# Remove rows with missing values in TripPurpose, ReportedDuration, and age
data = data[(data['TripPurpose'] != -1) & (data['ReportedDuration'] != -1) & (data['age'] != -1)]

# Feature selection
X = data.drop(columns=['Choice'])  # Features
Y = data['Choice']  # Target variable

# Label encoding for the target variable
LE = preprocessing.LabelEncoder()
LE.fit(Y)
Y = LE.transform(Y)

# Splitting the dataset into train, validation, and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)

# Scaling the features
scaler = preprocessing.StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_val_scale = scaler.transform(X_val)
X_test_scale = scaler.transform(X_test)

# Print the shapes of the datasets to confirm the split
print("Train data shape:", X_train.shape, Y_train.shape)
print("Validation data shape:", X_val.shape, Y_val.shape)
print("Test data shape:", X_test.shape, Y_test.shape)


# In[ ]:




