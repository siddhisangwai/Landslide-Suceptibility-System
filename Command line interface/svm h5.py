import pandas as pd
from sklearn.svm import SVC
import joblib  # Import joblib for model saving

# Load the dataset
# Replace 'your_dataset.csv' with the actual filename
dataset = pd.read_csv('dataset.csv')

# Split the dataset into features (X) and target labels (y)
X = dataset.drop('landslide', axis=1)
y = dataset['landslide']

# Create an SVM classifier
svm_classifier = SVC(probability=True)  # Enabling probability estimates

# Train the classifier on the entire dataset
svm_classifier.fit(X, y)

# Save the trained model to a .h5 file using joblib
joblib.dump(svm_classifier, 'svm_classifier_model.h5')