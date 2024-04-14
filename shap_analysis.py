import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import os


path = os.getcwd() + '/data/ex4Data.csv'

db = pd.read_csv(path)

X = db.iloc[:, 4:]
Y = db['Class']

# label encoding is done as model accepts only numeric values
# so strings need to be converted into labels
LE = preprocessing.LabelEncoder()
LE.fit(Y)
Y = LE.transform(Y)

# splitting dataset into train, validation and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=1)

# feature_names = X_train.columns.tolist()

# datapoints also need to be scaled into dataset with mean 0 and std dev = 1
X_train_scale = preprocessing.scale(X_train)
X_val_scale = preprocessing.scale(X_val)
X_test_scale = preprocessing.scale(X_test)

# Output the number of data points in training, validation, and test dataset.
print("Datapoints in Training set:", len(X_train))
print("Datapoints in validation set:", len(X_val))
print("Datapoints in Test set:", len(X_test))

train_logreg = LogisticRegression(random_state=0, max_iter=220)
train_logreg.fit(X_train_scale, Y_train)

pred_logreg = train_logreg.predict(X_val_scale)
print("For Logistic Regression: ")
print(classification_report(Y_val, pred_logreg))
print("Accuracy of logistic regression on the initial data is: ", accuracy_score(pred_logreg, Y_val))

classes = LE.inverse_transform(np.arange(0, train_logreg.coef_.shape[0]))
for i in range(train_logreg.coef_.shape[0]):
    print(f"\nModel coefficients ({classes[i]}):")
    for j in range(X.shape[1]):
        print(X.columns[j], "=", train_logreg.coef_[i, j].round(5))

# shap.partial_dependence_plot(
#     "xmin",
#     train_logreg.predict,
#     X,
#     ice=False,
#     model_expected_value=True,
#     feature_expected_value=True
# )
