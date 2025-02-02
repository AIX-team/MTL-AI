from fastapi import APIRouter
from services.googleMap_service import get_location

router = APIRouter()

@router.get("/location")
async def locations(lat: float, lng: float ):
    return await get_location(lat, lng)
