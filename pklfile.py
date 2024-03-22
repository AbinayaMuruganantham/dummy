import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

# Load the dataset
data = pd.read_csv(r"diabetes.csv")

#data = pd.read_csv("C:\Users/abina\Downloads\dpcode\codefordesignproject\data\diabetes.csv")

# Split the dataset into features (X) and target variable (y)
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Create a decision tree classifier
model = DecisionTreeClassifier()

# Train the model
model.fit(X, y)

# Save the trained model to a .pkl file
joblib.dump(model, "diabetes_model.pkl")