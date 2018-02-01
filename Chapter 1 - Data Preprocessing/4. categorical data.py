# Categorical Data

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