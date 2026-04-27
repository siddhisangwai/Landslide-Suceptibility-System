import requests

# API URL
api_url = "https://archive-api.open-meteo.com/v1/archive"

# Parameters for the API request
params = {
    "latitude": 18.5204,
    "longitude": 73.8567,
    "start_date": "2022-01-01",
    "end_date": "2022-12-31",
    "daily": "precipitation_sum",
    "timezone": "Asia/Bangkok"
}

# Make the API request
response = requests.get(api_url, params=params)

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
