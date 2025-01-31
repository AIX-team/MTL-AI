from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from models.googleMap_model import Location
from repository.googleMap_repository import get_locations
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
templates = Jinja2Templates(directory="templates")

# 지역 
@router.get("/location", response_model=List[Location])
def read_location():
    return [
         # {"name" : "Location 1", "lat": 37.4783, "lng": 126.9512},
        # {"name" : "Location 2", "lat": 34.0522, "lng": -118.2437},
        # {"name" : "Location 3", "lat": 40.7128, "lng": -74.0060 },
    ]

    
# api요청
@router.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    api_key = os.getenv("API_KEY")
    return templates.TemplateResponse("index.html",{"request":request, "api_key":api_key})