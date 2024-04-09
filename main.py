import pandas as pd
import os
from data_preprocessing import cleanup_dataset
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


path = os.getcwd() + '/data/ModeChoiceOptima.txt'
data = pd.read_csv(path, delimiter='\t')

filtered_data = cleanup_dataset(data)

print(filtered_data.describe())

X = filtered_data.drop('Choice', axis=1)
y = filtered_data['Choice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

regression = LogisticRegression(random_state=0, max_iter=200).fit(X_train_scaled, y_train)
print(regression.score(X_test_scaled, y_test))
