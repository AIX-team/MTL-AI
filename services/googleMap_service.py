from repository.googleMap_repository import get_locations

async def get_location(lat: float, lng: float):
    location_data = await get_locations(lat, lng)
    return location_data
    