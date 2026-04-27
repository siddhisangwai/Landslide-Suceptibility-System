import pandas as pd
import joblib  # Import joblib for model deserialization

# Load the saved Random Forest model from the .h5 file
rf_classifier = joblib.load('rf_classifier_model.h5')

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
# Predict the probability of landslide
prediction_probabilities = rf_classifier.predict_proba(new_data)

# Display the prediction results
print("Prediction probabilities for new data:")
print(f"Probability of No Landslide: {prediction_probabilities[0][0]:.2f}")
print(f"Probability of Landslide: {prediction_probabilities[0][1]:.2f}")

