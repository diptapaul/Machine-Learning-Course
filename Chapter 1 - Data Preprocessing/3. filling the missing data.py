import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset to the code
dataset = pd.read_csv('Data.csv')

# iloc for making a matrix by defining row and columns
# iloc[row_start:less than row_end, col_start:less than col_end]

# iloc[:, :-1] means matrix of all row and all the column without last column
independent_variable_x = dataset.iloc[:, :-1].values

# iloc[:, 3] means matrix of all row and 3 no. columns
dependent_variable_y = dataset.iloc[:, 3].values

# Taking the care of missing data

# Importing Importer class from scikit-learn library to work with missing data
from sklearn.preprocessing import Imputer

# Making an object of Importer class. Here parameters NaN is used to find out the missing values, strategy is used to fill the value, axis is used to calculate the value using column or row.
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

# fit method is used to fill the missing values to independent_variable_x matrix in imputer
imputer = imputer.fit(independent_variable_x[:, 1:3])

# transform is used to transform imputer and fill the missing values to the independent_variable_x matrix
independent_variable_x[:, 1:3] = imputer.transform(independent_variable_x[:, 1:3])