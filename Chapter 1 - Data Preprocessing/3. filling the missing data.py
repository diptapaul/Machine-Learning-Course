# Taking the care of missing data

# Importing Importer class from scikit-learn library to work with missing data
from sklearn.preprocessing import Imputer

# Making an object of Importer class. Here parameters NaN is used to find out the missing values, strategy is used to fill the value, axis is used to calculate the value using column or row.
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

# fit method is used to fill the missing values to independent_variable_x matrix in imputer
imputer = imputer.fit(independent_variable_x[:, 1:3])

# transform is used to transform imputer and fill the missing values to the independent_variable_x matrix
independent_variable_x[:, 1:3] = imputer.transform(independent_variable_x[:, 1:3])