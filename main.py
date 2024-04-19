import os
import warnings
from sklearn.exceptions import ConvergenceWarning

import pandas as pd
import data_preprocessing as dp
import data_modelling as dm
import data_analysis as da
from sklearn.linear_model import LogisticRegressionCV


# Suppresses convergence warnings from sag and saga solvers when testing solvers
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Parameters
path = os.getcwd() + '/data/ModeChoiceOptima.txt'  # Data set path
class_names = ['Public Transports', 'Private Modes', 'Soft Modes']  # Classification names
test_solvers = False  # If solvers for logistic regression should be tested

# Read in data set as pandas DataFrame
data = pd.read_csv(path, delimiter='\t')

# Clean the data set using the data_preprocessing cleanup method
filtered_data = dp.cleanup_dataset(data)

# Analyse cleaned up data
da.data_correlation('data/filtered_file.csv', 0.6)  # Generates correlation matrix
dp.descr_data(filtered_data)  # Describes data

# Split data in train and test sets and scales it
X, X_train, X_test, X_scaled, X_train_scaled, X_test_scaled, Y, Y_train, Y_test = dm.split_data(filtered_data)

# If testing different logistic regression solvers, creates confusion matrices for each solver
if test_solvers:
    dm.compare_lr_solvers(X_scaled, X_test_scaled, Y, Y_test, class_names)

# Create logistic regression model based on scaled data and cross-fold validation hyperparameter tuning
model = LogisticRegressionCV(random_state=0, max_iter=200, class_weight='balanced', cv=5).fit(X_scaled, Y)

# Check model performance using confusion matrices
da.confusion_matrices(model, X_test_scaled, Y_test, class_names)

# Calculate SHAP values using the model and the data
shap_values = dm.shap_val(model, X_train_scaled, X, X_test, X_test_scaled)

# Plot the SHAP values as bar, beeswarm, and scatter plots
da.shap_plot(shap_values, class_names, 'bar')
da.shap_plot(shap_values, class_names, 'beeswarm')
da.shap_plot(shap_values, class_names, 'scatter')
da.shap_sum_bar_plot(shap_values, class_names)
