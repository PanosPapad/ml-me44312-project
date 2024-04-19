from typing import Any
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegressionCV
from data_analysis import confusion_matrices


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, Any, Any, Any, Any, Any, Any, Any, Any]:
    # Feature selection
    X = data.drop(columns=['Choice'])  # Features
    Y = data['Choice']  # Target variable

    # Splitting the dataset into train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Scaling the features
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Print the shapes of the datasets to confirm the split
    print("Train data shape:", X_train.shape, Y_train.shape)
    print("Test data shape:", X_test.shape, Y_test.shape)

    return X, X_train, X_test, X_scaled, X_train_scaled, X_test_scaled, Y, Y_train, Y_test


def compare_lr_solvers(X_scaled: pd.DataFrame, X_test_scaled: pd.DataFrame, Y: pd.Series, Y_test: pd.Series,
                       class_names: list[str], save: bool = False) -> None:
    # Other solves used in comparison to default solver
    other_solvers = ['newton-cg', 'sag', 'saga']

    # Generate models with the solvers
    lbfgs = LogisticRegressionCV(random_state=0, max_iter=200, class_weight='balanced', cv=5).fit(X_scaled, Y)
    newton = LogisticRegressionCV(random_state=0, max_iter=200, class_weight='balanced', cv=5,
                                  solver=other_solvers[0]).fit(X_scaled, Y)
    sag = LogisticRegressionCV(random_state=0, max_iter=800, class_weight='balanced', cv=5,
                               solver=other_solvers[1]).fit(X_scaled, Y)
    saga = LogisticRegressionCV(random_state=0, max_iter=800, class_weight='balanced', cv=5,
                                solver=other_solvers[2]).fit(X_scaled, Y)

    # For each model and solver name, compute confusion matrices
    solvers = [(lbfgs, 'lbfgs'), (newton, other_solvers[0]), (sag, other_solvers[1]), (saga, other_solvers[2])]
    for solver in solvers:
        confusion_matrices(solver[0], X_test_scaled, Y_test, class_names, save, solver[1])


def shap_val(model: Any, mask: pd.DataFrame, X: pd.DataFrame, X_test: pd.DataFrame, X_test_scaled: pd.DataFrame) \
        -> shap.Explanation:
    """
    Generates the SHAP values for a given linear model and features.

    :param model:         Linear machine learning model that needs explaining using SHAP values.
    :param mask:          Train set for the SHAP explainer, should be scaled.
    :param X:             All features.
    :param X_test:        The features of the test set.
    :param X_test_scaled: The scaled features of the test set.
    :return:              SHAP values as an Explanation object.
    """
    # Set the masker or train data for the SHAP explainer
    masker = shap.maskers.Independent(data=mask)

    # Generate SHAP explainer and values
    explainer = shap.LinearExplainer(model, masker=masker)
    shap_values = explainer(X_test_scaled)

    # Add back the feature names stripped by the StandardScaler
    for i, c in enumerate(X.columns):
        shap_values.feature_names[i] = c

    # Convert back to the original data
    shap_values.data = X_test.values

    return shap_values
