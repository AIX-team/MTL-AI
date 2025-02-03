from pydantic import BaseModel

class MapLocation(BaseModel):
    name: str
    lat: float
    lng: float
   