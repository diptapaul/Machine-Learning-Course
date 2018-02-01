import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Data.csv')

independent_variable_x = dataset.iloc[:, :-1].values
dependent_variable_y = dataset.iloc[:, 3].values

X_train, X_test, y_train, y_test = train_test_split(independent_variable_x, dependent_variable_y, test_size = 0.2, random_state = 0)

"""
X_sc = StandardScaler();
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.transform(X_test) 
"""