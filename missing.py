import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer

# Example data for X and y
X = np.array([[30, 2], [6, 150], [7, np.nan]])  # Input features with missing values
y = np.array([0, 1, 1])  # Target variable

# Replace the missing values with np.nan
X[X == 0] = np.nan

# Create an imputer to fill in missing values with the mean
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform it
X_imputed = imputer.fit_transform(X)

# Create and train the DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_imputed, y)
