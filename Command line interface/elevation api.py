import requests

# API URL
api_url = "https://api.open-meteo.com/v1/elevation"

# Parameters for the API request
params = {
    "latitude": 18.5204,
    "longitude": 73.8567,
}

# Make the API request
response = requests.get(api_url, params=params)

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
