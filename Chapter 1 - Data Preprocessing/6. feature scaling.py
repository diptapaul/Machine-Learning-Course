# Feature Scaling

# Importing StandardScaler class from scikit learn library
from sklearn.preprocessing import StandardScaler

# Making an object of StandardScaler class
X_sc = StandardScaler();

# Feater Scaling the X_train variable. It needs to be fitted before transforming it.
X_train = X_sc.fit_transform(X_train)

# Feater Scaling the X_train variable. It needs to be fitted before transforming it.
X_test = X_sc.transform(X_test)