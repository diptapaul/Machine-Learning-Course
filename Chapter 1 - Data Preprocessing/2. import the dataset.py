# Importing the dataset to the code
dataset = pd.read_csv('Data.csv')

# iloc for making a matrix by defining row and columns
# iloc[row_start:row_end, col_start:col_end]

# iloc[:, :-1] means matrix of all row and all the column without last column
independent_variable_x = dataset.iloc[:, :-1].values

#iloc[:, 3] means matrix of all row and 3 no. columns
dependent_variable_y = dataset.iloc[:, 3].values