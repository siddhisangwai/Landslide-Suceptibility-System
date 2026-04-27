import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename
dataset = pd.read_csv('dataset.csv')

# Split the dataset into features (X) and target labels (y)
X = dataset.drop('landslide', axis=1)
y = dataset['landslide']

# Create a Logistic Regression classifier
log_reg = LogisticRegression()

# Train the classifier on the entire dataset
log_reg.fit(X, y)

# Save the trained model to a .h5 file
model_filename = 'logistic_regression_model.h5'
joblib.dump(log_reg, model_filename)

print("Model trained and saved to", model_filename)