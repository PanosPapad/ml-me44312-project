import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

path = os.getcwd() + '/data/ModeChoiceOptima.txt'
data = pd.read_csv(path, delimiter='\t')

# print(data.head())
print('---')
print(data.describe())

numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns

# # Uncomment to plot histograms for all columns with numerical values
# # DO NOT run without a reason. Expensive operation
# for col in numerical_cols:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data[col], kde=True)
#     plt.title(f'Histogram of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.show()
