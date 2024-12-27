import requests
import json
from datetime import datetime, timedelta


# Function to fetch the latest data for a specific token
def fetch_latest_data(token_address):
    url = f"https://api.dexscreener.com/latest/dex/tokens/{token_address}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()  # Return the JSON response
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
    return None


# Function to fetch data for the last month
def fetch_data_for_last_month(token_address):
    # Get today's date
    end_date = datetime.now()
    # Get the date one month ago
    start_date = end_date - timedelta(days=30)

    # Loop through each day in the last month
    current_date = start_date
    while current_date <= end_date:
        data = fetch_latest_data(token_address)
        if data:
            # Process the data as needed
            print(f"Data for {current_date.strftime('%Y-%m-%d')}:")
            print(json.dumps(data, indent=4))  # Pretty print the JSON response
        else:
            print(f"No data available for {current_date.strftime('%Y-%m-%d')}.")
        current_date += timedelta(days=1)  # Move to the next day


# Example token address (replace with the actual token address you want to query)
token_address = "v3zjmp8c1kzgk8oo6bwvuyrqx57mvwy13hhop6uy7v6"

# Fetch data for the last month
fetch_data_for_last_month(token_address)