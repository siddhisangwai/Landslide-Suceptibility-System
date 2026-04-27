import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Load the trained model
model = keras.models.load_model('cnn_model.h5')

# Load the training data to fit the scaler
training_data = pd.read_csv('dataset.csv')
X_train = training_data.drop('landslide', axis=1)

# Fit the StandardScaler on the training data
scaler = StandardScaler()
scaler.fit(X_train)

# Define a function to predict landslide probabilities for new data
def predict_landslide_probability(new_data, model, scaler):
    # Scale the new data using the fitted scaler
    new_data_scaled = scaler.transform(new_data)

    # Predict probabilities for the new data
    prediction_probabilities = model.predict(new_data_scaled)

    return prediction_probabilities

# Load elevation data from the file
with open("elevation_data.txt", "r") as elevation_file:
    elevation_value = int(elevation_file.read())

with open("rainfall_data.txt", "r") as rainfall_file:
    rainfall_value = int(rainfall_file.read())

with open("precipitation_data.txt", "r") as precipitation_file:
    precipitation_value = int(precipitation_file.read())

# Load soil type from the file
with open("soil_type.txt", "r") as soil_type_file:
    soil_type = soil_type_file.read().strip()  # Read and remove leading/trailing whitespace

# Example prediction using new data, including elevation and soil type
new_data = pd.DataFrame({
    'slope': [9],
    'precipitation': [precipitation_value],
    'elevation': [elevation_value],  # Use the elevation data from the file
    'soil_type': [soil_type],  # Use the soil type obtained from the file
    'rainfall': [rainfall_value]
})

predicted_probabilities = predict_landslide_probability(new_data, model, scaler)

print("Prediction probabilities for new data:")
print(f"Probability of No Landslide: {predicted_probabilities[0][0]:.2f}")
print(f"Probability of Landslide: {predicted_probabilities[0][1]:.2f}")
