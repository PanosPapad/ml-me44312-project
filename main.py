import pandas as pd
import os
from data_preprocessing import cleanup_dataset


path = os.getcwd() + '/data/ModeChoiceOptima.txt'
data = pd.read_csv(path, delimiter='\t')

cleanup_dataset(data)
