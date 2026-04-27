import subprocess
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import requests
import http.server
import socketserver
from urllib.parse import parse_qs
import numpy as np
import re
from tensorflow import keras
from keras.preprocessing.image import img_to_array, load_img

# Load machine learning models
model = keras.models.load_model('my_model.h5')
svm_model = joblib.load('svm_classifier_model.h5')
lr_model = joblib.load('logistic_regression_model.h5')
rf_model = joblib.load('rf_classifier_model.h5')
cnn_model = keras.models.load_model('cnn_model.h5')

# ... (other imports and model loading) ...

# Define API URLs and parameters
elevation_api_url = "https://api.open-meteo.com/v1/elevation"
rainfall_api_url = "https://archive-api.open-meteo.com/v1/archive"
precipitation_api_url = "https://archive-api.open-meteo.com/v1/archive"

# Define a custom request handler to process form data with file upload
class MyRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
     if self.path == '/upload':
        content_type, _ = self.headers.get('Content-Type').split(';')
        if content_type == 'multipart/form-data':
            content_length = int(self.headers.get('Content-Length'))
            boundary = self.headers.get('Content-Type').split('=')[1].encode()

            # Read the binary file data from the request
            file_data = self.rfile.read(content_length)

            # Split the file data into parts using the boundary
            parts = file_data.split(b'--' + boundary)

            for part in parts:
                if b'filename="' in part:
                    # This part contains the uploaded file
                    # Use regular expressions to extract the filename
                    filename_match = re.search(rb'filename="(.+)"', part)
                    if filename_match:
                        filename = filename_match.group(1)
                        with open('image.jpg', 'wb') as f:
                            f.write(part.split(b'\r\n\r\n')[1])
                        
                        # Send a JavaScript response to update the result
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        
                        response = b'<script type="text/javascript">parent.handleUploadResponse("Image uploaded successfully as image.jpg");</script>'
                        self.wfile.write(response)
                        
                        return

        self.send_response(400)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        # Send a JavaScript response to update the result with an error message
        response = b'<script type="text/javascript">parent.handleUploadResponse("Invalid content type");</script>'
        self.wfile.write(response)
     elif self.path == '/process':
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length).decode('utf-8')
        post_params = parse_qs(post_data)

        if 'slope' in post_params and 'latitude' in post_params and 'longitude' in post_params:
            try:
                latitude = float(post_params['latitude'][0])
                longitude = float(post_params['longitude'][0])
                slope = int(post_params['slope'][0])

                # Elevation API request
                elevation_params = {
                    "latitude": latitude,
                    "longitude": longitude,
                }
                elevation_response = requests.get(elevation_api_url, params=elevation_params)

                # Rainfall API request
                rainfall_params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": "2023-09-24",
                    "end_date": "2023-11-24",
                    "daily": "rain_sum",
                    "timezone": "Asia/Bangkok"
                }
                rainfall_response = requests.get(rainfall_api_url, params=rainfall_params)

                # Precipitation API request
                precipitation_params = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "start_date": "2023-09-24",
                    "end_date": "2023-11-24",
                    "daily": "precipitation_sum",
                    "timezone": "Asia/Bangkok"
                }
                precipitation_response = requests.get(precipitation_api_url, params=precipitation_params)

                if (
                    elevation_response.status_code == 200
                    and rainfall_response.status_code == 200
                    and precipitation_response.status_code == 200
                ):
                    elevation_data = elevation_response.json().get("elevation", [0])
                    elevation = int(elevation_data[0])

                    rain_sum_data = rainfall_response.json().get("daily", {}).get("rain_sum", [])
                    filtered_rain_data = [value for value in rain_sum_data if value is not None]
                    total_rainfall = int(round(sum(filtered_rain_data, 0)))

                    precipitation_sum_data = precipitation_response.json().get("daily", {}).get("precipitation_sum", [])
                    filtered_preci_data = [value for value in precipitation_sum_data if value is not None]
                    total_precipitation = int(round(sum(filtered_preci_data, 0)))


                    image_path = "image.jpg"
                    target_size = (220, 220)
                    img = load_img(image_path, target_size=target_size)
                    preprocessed_input = img_to_array(img) / 255.0
                    predictions = model.predict(np.expand_dims(preprocessed_input, axis=0))
                    predicted_class = np.argmax(predictions)
                    class_labels = ['2', '5', '3', '4', '1']
                    soil_type_mapping = {
                    '1': 'yello soil',
                    '2': 'black soil',
                    '3': 'laterite soil',
                    '4': 'peat soil',
                    '5': 'cinder soil'
                     }
                    predicted_label = class_labels[predicted_class]
                    predicted_soil_type = soil_type_mapping[predicted_label]

                    # Example prediction using new data, including elevation and soil type
                    new_data = pd.DataFrame({
                        'slope': [slope],
                        'precipitation': [total_precipitation],
                        'elevation': [elevation],  # Use the elevation data from the file
                        'soil_type': [predicted_label],  # Use the soil type obtained from the file
                        'rainfall': [total_rainfall]
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


                    # ... (other model predictions and response HTML) ...
                    response_html = f"""
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>Landslide Prediction</title>
                    </head>
                    <body>
                        <h2>Coordinates: Latitude {latitude}, Longitude {longitude}</h2>
                        <p>Slope: {slope} degrees</p>
                        <p>Elevation: {elevation} meters</p>
                        <p>Total Rainfall: {total_rainfall} mm</p>
                        <p>Total Precipitation: {total_precipitation} mm</p>
                        <p>Soil Type: {predicted_soil_type}</p>
                        <h3>Landslide Probability</h3>
                        <p>Average Probability of No Landslide: {average_probabilities[0][1]:.2f}</p>
                        <p>Average Probability of Landslide: {average_probabilities[0][0]:.2f}</p>
                    </body>
                    </html>
                    """

                    # Send the response to the client
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(response_html.encode('utf-8'))
                else:
                    self.send_response(500)
                    self.end_headers()
            except ValueError:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(400)
            self.end_headers()

# Create a simple web server
with socketserver.TCPServer(("", 8000), MyRequestHandler) as httpd:
    print("Server is running on http://localhost:8000")
    httpd.serve_forever()