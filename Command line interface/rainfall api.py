import requests

# API URL
api_url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters for the API request
params = {
    "latitude": 18.5204,
    "longitude": 73.8567,
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "daily": "rain_sum",
    "timezone": "Asia/Bangkok"
}

# Make the API request
response = requests.get(api_url, params=params)

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
