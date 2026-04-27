import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib  # Import joblib for model serialization

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename
dataset = pd.read_csv('dataset.csv')

# Split the dataset into features (X) and target labels (y)
X = dataset.drop('landslide', axis=1)
y = dataset['landslide']

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the entire dataset
rf_classifier.fit(X, y)

# Save the trained model to a .h5 file
joblib.dump(rf_classifier, 'rf_classifier_model.h5')