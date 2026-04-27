import subprocess
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import requests

# API URL for elevation data
elevation_api_url = "https://api.open-meteo.com/v1/elevation"

# Parameters for elevation API request
elevation_params = {
    "latitude": 18.5204,
    "longitude": 73.8567,
}

# API URL for daily precipitation data
precipitation_api_url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters for daily precipitation API request
precipitation_params = {
    "latitude": 18.5204,
    "longitude": 73.8567,
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "daily": "precipitation_sum",
    "timezone": "Asia/Bangkok"
}

# API URL for daily rainfall data
rainfall_api_url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters for daily rainfall API request
rainfall_params = {
    "latitude": 18.5204,
    "longitude": 73.8567,
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "daily": "rain_sum",
    "timezone": "Asia/Bangkok"
}

# Make the API request
response = requests.get(elevation_api_url, params=elevation_params)

if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract the elevation value
    elevation_data = data.get("elevation", [])

    if elevation_data:
        elevation = int(elevation_data[0])  # Convert to integer
        print("Elevation:", elevation, "meters")

        # Save the elevation data to a file
        with open("elevation_data.txt", "w") as elevation_file:
            elevation_file.write(str(elevation))
    else:
        print("Elevation data not found in the response.")
else:
    print("Failed to fetch data from the API. Status code:", response.status_code)

# Make the API request
response = requests.get(rainfall_api_url, params=rainfall_params)

if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract the "rain_sum" data
    rain_sum_data = data.get("daily", {}).get("rain_sum")

    filtered_rain_data = [value for value in rain_sum_data if value is not None]
    
    if filtered_rain_data:
        total_rainfall = sum(filtered_rain_data)
        total_rainfall_int = int(round(total_rainfall))  # Round and convert to integer
        print("Total Rainfall:", total_rainfall_int, "mm")

        with open("rainfall_data.txt", "w") as rainfall_file:
            rainfall_file.write(str(total_rainfall_int))        
    else:
        print("Rain Sum data not found in the response.")
else:
    print("Failed to fetch data from the API. Status code:", response.status_code)

# Make the API request
response = requests.get(precipitation_api_url, params=precipitation_params)

if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Extract the "precipitation_sum" data
    rain_sum_data = data.get("daily", {}).get("precipitation_sum")

    filtered_preci_data = [value for value in rain_sum_data if value is not None]
    
    if filtered_preci_data:
        total_precipitation = sum(filtered_preci_data)
        total_precipitation_int = int(round(total_precipitation))  # Round and convert to integer
        print("Total Precipitation:", total_precipitation_int, "mm")

        with open("precipitation_data.txt", "w") as precipitation_file:
            precipitation_file.write(str(total_precipitation_int)) 
    else:
        print("Rain Sum data not found in the response.")
else:
    print("Failed to fetch data from the API. Status code:", response.status_code)

subprocess.run(["python", "soil type.py"])


# Load the saved SVM classifier model
svm_model = joblib.load('svm_classifier_model.h5')

# Load the saved Logistic Regression model
lr_model = joblib.load('logistic_regression_model.h5')

# Load the saved Random Forest model
rf_model = joblib.load('rf_classifier_model.h5')

# Load the trained CNN model
cnn_model = keras.models.load_model('cnn_model.h5')

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

# Predict on the new data using each model
svm_probabilities = svm_model.predict_proba(new_data)
lr_probabilities = lr_model.predict_proba(new_data)
rf_probabilities = rf_model.predict_proba(new_data)

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

cnn_probabilities = predict_landslide_probability(new_data, cnn_model, scaler)

# Calculate the average of the predicted probabilities
average_probabilities = (svm_probabilities + lr_probabilities + rf_probabilities + cnn_probabilities) / 4

# Display the average probabilities
print("Average Prediction Probabilities for new data:")
print(f"Average Probability of No Landslide: {average_probabilities[0][0]:.2f}")
print(f"Average Probability of Landslide: {average_probabilities[0][1]:.2f}")
