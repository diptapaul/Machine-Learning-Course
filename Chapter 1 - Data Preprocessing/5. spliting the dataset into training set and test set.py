import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data to the code
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

# LabelEncoder is a class which is used to label string values to a integer value
# OneHotEncoder is a class and used for create dumb variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Making an object of LabelEncoder class
labelencoder_x = LabelEncoder()

# fit_transform is used to transform the value of 0 no. column to integer and fit it to the object and assignning it to the independent_variable_x matrix
independent_variable_x[:, 0] = labelencoder_x.fit_transform(independent_variable_x[:, 0])

# Making an object using 0 no. index of dataset
onehotencoder = OneHotEncoder(categorical_features = [0])

# Assing the fited & transformed data to the independent_variable_x
independent_variable_x = onehotencoder.fit_transform(independent_variable_x).toarray()

# Making an object of LabelEncoder class
labelencoder_y = LabelEncoder()

# fit_transform is used to transform the value of 0 no. column to integer and fit it to the object and assignning it to the dependent_variable_y matrix
dependent_variable_y = labelencoder_y.fit_transform(dependent_variable_y)

# Spliting the dataset into Training set and Test set

# Importing the library train_test_split from scikit-learn library
from sklearn.cross_validation import train_test_split

# Here test_size = 0.2 to split 20% data to the Test set and rest 80% to the Training set.
# random_state = 0 is used to sampling the data randomly. Another good choice is random_state = 42
X_train, X_test, y_train, y_test = train_test_split(independent_variable_x, dependent_variable_y, test_size = 0.2, random_state = 0)