import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("api_key")

async def fetch_place(placeId):
    url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={placeId}&key={API_KEY}"
    response = requests.get(url)
    return response.json()
