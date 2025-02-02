import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("api_key")

async def get_locations(lat: float, lng: float):
    url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={API_KEY}"
    response = requests.get(url)
    return response.json()