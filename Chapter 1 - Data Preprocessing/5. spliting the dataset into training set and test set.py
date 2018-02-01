# Spliting the dataset into Training set and Test set

# Importing the library train_test_split from scikit-learn library
from sklearn.cross_validation import train_test_split

# Here test_size = 0.2 to split 20% data to the Test set and rest 80% to the Training set.
# random_state = 0 is used to sampling the data randomly. Another good choice is random_state = 42
X_train, X_test, y_train, y_test = train_test_split(independent_variable_x, dependent_variable_y, test_size = 0.2, random_state = 0)